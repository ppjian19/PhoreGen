import numpy as np
import copy
import torch
from torch.nn import functional as F
from rdkit.Chem import AllChem as Chem
from rdkit import Geometry
from rdkit import RDLogger
from openbabel import openbabel as ob
import itertools
import re
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from utils.predict_bonds import predict_bonds

# ATOM_TYPES = [0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
ATOM_TYPES = [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # removed masked_atom


def get_fully_connected_edge(num_nodes):
    full_src = torch.repeat_interleave(torch.arange(num_nodes), num_nodes)
    full_dst = torch.arange(num_nodes).repeat(num_nodes)
    # mask = full_dst != full_src
    # full_dst, full_src = full_dst[mask], full_src[mask]
    return torch.stack([full_src, full_dst], dim=0)


def sample_from_interval(lower, upper, batch_size, mode='uniform', scale=4.0):
    if mode == 'uniform':
        num_nodes = torch.randint(lower, upper + 1, (batch_size,))
    elif mode == 'normal':
        mid = (lower + upper) / 2
        std_dev = (upper - lower) / scale
        num_nodes = torch.normal(mid, std_dev, (batch_size,)).clamp(lower, upper).round().int()
    else:
        raise NotImplementedError (f"The sample nodes mode {mode} is not implemented.")
    return num_nodes


def make_edge_data(num_atoms, device=None):
    if device is None:
        device = num_atoms.device
    edge_index = []
    edge_batch = []
    idx_start = 0
    for idx, n_nodes in enumerate(num_atoms):
        halfedge_index = torch.triu_indices(n_nodes, n_nodes, offset=1).to(device)
        fulledge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        edge_index.append(fulledge_index + idx_start)
        edge_batch.append(torch.full((fulledge_index.shape[1],), idx))
        idx_start += n_nodes
    edge_index = torch.cat(edge_index, dim=1).to(device)
    edge_batch = torch.cat(edge_batch, dim=0).to(device)
    return edge_index, edge_batch


def unbatch_data(results, n_graphs, include_bond=True):
    outputs_pred = results['pred']
    outputs_traj = results['traj']
    batch_node = results['lig_info'][1]
    edge_index = results['lig_info'][2]
    batch_edge = results['lig_info'][3]

    new_outputs = []
    for i_mol in range(n_graphs):
        ind_node = (batch_node == i_mol)
        ind_edge = (batch_edge == i_mol)
        assert ind_node.sum() * (ind_node.sum()-1) == ind_edge.sum()
        if include_bond:
            new_pred_this = [outputs_pred[0][ind_node],  # node type
                            outputs_pred[1][ind_node],  # node pos
                            outputs_pred[2][ind_edge]]  # halfedge type
            
            new_traj_this = [outputs_traj[0][:, ind_node],  # node type. The first dim is time
                            outputs_traj[1][:, ind_node],  # node pos
                            outputs_traj[2][:, ind_edge]]  # halfedge type
        else:
            new_pred_this = [outputs_pred[0][ind_node],  # node type
                            outputs_pred[1][ind_node]]  # node pos
            
            new_traj_this = [outputs_traj[0][:, ind_node],  # node type. The first dim is time
                            outputs_traj[1][:, ind_node]]  # node pos

        edge_index_this = edge_index[:, ind_edge]
        assert ind_node.nonzero()[0].min() == edge_index_this.min()
        edge_index_this = edge_index_this - ind_node.nonzero()[0].min()

        new_outputs.append({
            'pred': new_pred_this,
            'traj': new_traj_this,
            'edge_index': edge_index_this,
        })
    return new_outputs


def decode_data(pred_info, edge_index, include_bond=True, num_bond_types=5):
    pred_node=pred_info[0]
    pred_pos=pred_info[1]
    atom_type = F.softmax(pred_node, dim=-1).argmax(dim=-1)
    isnot_masked_atom = (atom_type < len(ATOM_TYPES))  # 11 types (removed masked atoms)
    
    if not isnot_masked_atom.all():
        # masked atom detected
        edge_index_changer = - torch.ones(len(isnot_masked_atom), dtype=torch.long)
        edge_index_changer[isnot_masked_atom] = torch.arange(isnot_masked_atom.sum())
    
    atom_type = atom_type[isnot_masked_atom]
    element = [ATOM_TYPES[i] for i in atom_type]
   
    atom_pos = pred_pos[isnot_masked_atom]
    
    if include_bond:
        pred_edge=pred_info[2]
        edge_type = F.softmax(pred_edge, dim=-1).argmax(dim=-1)
        is_bond = (edge_type > 0) & (edge_type < num_bond_types)  # 0, 1, 2, 3, 4
        bond_type = edge_type[is_bond]
        bond_index = edge_index[:, is_bond]
        if not isnot_masked_atom.all():
            # masked atom detected
            bond_index = edge_index_changer[bond_index]
            bond_for_masked_atom = (bond_index < 0).any(dim=0)
            bond_index = bond_index[:, ~bond_for_masked_atom]
            bond_type = bond_type[~bond_for_masked_atom]
    else:
        bond_type, bond_index = None, None

    return {
        'element': element,
        'atom_pos': atom_pos,
        'bond_type': bond_type,
        'bond_index': bond_index,
    }


def compute_atom_prox_loss(pos, edge_index, edge_type, min_d=1.2, max_d=2.8):
    if torch.any(edge_type > 0):
        bond_index = edge_index[:, edge_type > 0]
        bond_len = torch.norm(pos[bond_index[0]] - pos[bond_index[1]], dim=-1)
        loss = torch.mean(torch.clamp(bond_len - max_d, min=0) + torch.clamp(min_d - bond_len, min=0))
    else:
        loss = torch.tensor(0., device=pos.device)
    return loss


def compute_batch_atom_prox_loss(pred_ligand_pos, ligand_batch, pred_edge, 
                    ligand_edge_index, ligand_edge_batch, min_d=1.2, max_d=2.8):
    batch_losses = torch.tensor(0., device=pred_ligand_pos.device)
    num_graphs = ligand_batch.max().item() + 1
    for i in range(num_graphs):
        pos = pred_ligand_pos[ligand_batch == i]
        edge_index = ligand_edge_index[:, ligand_edge_batch == i] - (ligand_batch == i).nonzero()[0].min()
        edge_type = pred_edge[ligand_edge_batch == i].argmax(-1)
        batch_losses += compute_atom_prox_loss(pos, edge_index, edge_type, min_d=min_d, max_d=max_d)
    # return batch_losses
    return batch_losses / num_graphs


def compute_batch_center_prox_loss(pred_ligand_pos, ligand_batch, phore_center):
    batch_losses = torch.tensor(0., device=pred_ligand_pos.device)
    num_graphs = ligand_batch.max().item() + 1
    for i in range(num_graphs):
        pos = pred_ligand_pos[ligand_batch == i]
        batch_losses += torch.norm(pos.mean(dim=0) - phore_center, p=2, dim=-1)
    # return batch_losses
    return batch_losses / num_graphs


"""
https://github.com/mattragoza/liGAN/blob/master/fitting.py

License: GNU General Public License v2.0
https://github.com/mattragoza/liGAN/blob/master/LICENSE
"""

class MolReconsError(Exception):
    pass



def reachable_r(a,b, seenbonds):
    '''Recursive helper.'''

    for nbr in ob.OBAtomAtomIter(a):
        bond = a.GetBond(nbr).GetIdx()
        if bond not in seenbonds:
            seenbonds.add(bond)
            if nbr == b:
                return True
            elif reachable_r(nbr,b,seenbonds):
                return True
    return False


def reachable(a,b):
    '''Return true if atom b is reachable from a without using the bond between them.'''
    if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
        return False #this is the _only_ bond for one atom
    #otherwise do recursive traversal
    seenbonds = set([a.GetBond(b).GetIdx()])
    return reachable_r(a,b,seenbonds)


def forms_small_angle(a,b,cutoff=45):
    '''Return true if bond between a and b is part of a small angle
    with a neighbor of a only.'''

    for nbr in ob.OBAtomAtomIter(a):
        if nbr != b:
            degrees = b.GetAngle(a,nbr)
            if degrees < cutoff:
                return True
    return False


def count_nbrs_of_elem(atom, atomic_num):
    '''
    Count the number of neighbors atoms
    of atom with the given atomic_num.
    '''
    count = 0
    for nbr in ob.OBAtomAtomIter(atom):
        if nbr.GetAtomicNum() == atomic_num:
            count += 1
    return count


def connect_the_dots(mol, atoms, maxbond=4):
    '''Custom implementation of ConnectTheDots.  This is similar to
    OpenBabel's version, but is more willing to make long bonds 
    (up to maxbond long) to keep the molecule connected.  It also 
    attempts to respect atom type information from struct.
    atoms and struct need to correspond in their order
    Assumes no hydrogens or existing bonds.
    '''
    pt = Chem.GetPeriodicTable()

    if len(atoms) == 0:
        return

    mol.BeginModify()

    #just going to to do n^2 comparisons, can worry about efficiency later
    coords = np.array([(a.GetX(),a.GetY(),a.GetZ()) for a in atoms])
    dists = squareform(pdist(coords))
    # types = [struct.channels[t].name for t in struct.c]

    for (i,a) in enumerate(atoms):
        for (j,b) in enumerate(atoms):
            if a == b:
                break
            if dists[i,j] < 0.01:  #reduce from 0.4
                continue #don't bond too close atoms
            if dists[i,j] < maxbond:
                flag = 0
                # if indicators[i][ATOM_FAMILIES_ID['Aromatic']] and indicators[j][ATOM_FAMILIES_ID['Aromatic']]:
                    # print('Aromatic', ATOM_FAMILIES_ID['Aromatic'], indicators[i])
                    # flag = ob.OB_AROMATIC_BOND
                # if 'Aromatic' in types[i] and 'Aromatic' in types[j]:
                #     flag = ob.OB_AROMATIC_BOND
                mol.AddBond(a.GetIdx(),b.GetIdx(),1,flag)

    atom_maxb = {}
    for (i,a) in enumerate(atoms):
        #set max valance to the smallest max allowed by openbabel or rdkit
        #since we want the molecule to be valid for both (rdkit is usually lower)
        maxb = ob.GetMaxBonds(a.GetAtomicNum())
        maxb = min(maxb,pt.GetDefaultValence(a.GetAtomicNum())) 

        if a.GetAtomicNum() == 16: # sulfone check
            if count_nbrs_of_elem(a, 8) >= 2:
                maxb = 6

        # if indicators[i][ATOM_FAMILIES_ID['Donor']]:
        #     maxb -= 1 #leave room for hydrogen
        # if 'Donor' in types[i]:
        #     maxb -= 1 #leave room for hydrogen
        atom_maxb[a.GetIdx()] = maxb
    
    #remove any impossible bonds between halogens
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
            mol.DeleteBond(bond)

    def get_bond_info(biter):
        '''Return bonds sorted by their distortion'''
        bonds = [b for b in biter]
        binfo = []
        for bond in bonds:
            bdist = bond.GetLength()
            #compute how far away from optimal we are
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum()) 
            stretch = bdist-ideal
            binfo.append((stretch,bdist,bond))
        binfo.sort(reverse=True, key=lambda t: t[:2]) #most stretched bonds first
        return binfo

    #prioritize removing hypervalency causing bonds, do more valent 
    #constrained atoms first since their bonds introduce the most problems
    #with reachability (e.g. oxygen)
    # hypers = sorted([(atom_maxb[a.GetIdx()],a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms],key=lambda aa: (aa[0],-aa[1]))
    # for mb,diff,a in hypers:
    #     if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
    #         continue
    #     binfo = get_bond_info(ob.OBAtomBondIter(a))
    #     for stretch,bdist,bond in binfo:
    #         #can we remove this bond without disconnecting the molecule?
    #         a1 = bond.GetBeginAtom()
    #         a2 = bond.GetEndAtom()

    #         #get right valence
    #         if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or \
    #             a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
    #             #don't fragment the molecule
    #             if not reachable(a1,a2):
    #                 continue
    #             mol.DeleteBond(bond)
    #             if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
    #                 break #let nbr atoms choose what bonds to throw out


    binfo = get_bond_info(ob.OBMolBondIter(mol))
    #now eliminate geometrically poor bonds
    for stretch,bdist,bond in binfo:
        #can we remove this bond without disconnecting the molecule?
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        #as long as we aren't disconnecting, let's remove things
        #that are excessively far away (0.45 from ConnectTheDots)
        #get bonds to be less than max allowed
        #also remove tight angles, because that is what ConnectTheDots does
        if stretch > 0.45 or forms_small_angle(a1,a2) or forms_small_angle(a2,a1):
            #don't fragment the molecule
            if not reachable(a1,a2):
                continue
            mol.DeleteBond(bond)

    mol.EndModify()


def fixup(atoms, mol, ):
    '''Set atom properties to match channel.  Keep doing this
    to beat openbabel over the head with what we want to happen.'''

    mol.SetAromaticPerceived(True)  #avoid perception
    for i, atom in enumerate(atoms):
        # ch = struct.channels[t]
        # ind = indicators[i]

        # if ind[ATOM_FAMILIES_ID['Aromatic']]:
        #     atom.SetAromatic(True)
        #     atom.SetHyb(2)

        # if ind[ATOM_FAMILIES_ID['Donor']]:
        #     if atom.GetExplicitDegree() == atom.GetHvyDegree():
        #         if atom.GetHvyDegree() == 1 and atom.GetAtomicNum() == 7:
        #             atom.SetImplicitHCount(2)
        #         else:
        #             atom.SetImplicitHCount(1) 


        # elif ind[ATOM_FAMILIES_ID['Acceptor']]: # NOT AcceptorDonor because of else
        #     atom.SetImplicitHCount(0)   

        if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():     # Nitrogen, Oxygen
            #this is a little iffy, ommitting until there is more evidence it is a net positive
            #we don't have aromatic types for nitrogen, but if it
            #is in a ring with aromatic carbon mark it aromatic as well
            acnt = 0
            for nbr in ob.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    acnt += 1
            if acnt > 1:
                atom.SetAromatic(True)


def make_obmol(xyz, atomic_numbers):
    mol = ob.OBMol()
    mol.BeginModify()
    atoms = []
    for xyz,t in zip(xyz, atomic_numbers):
        x,y,z = xyz
        # ch = struct.channels[t]
        atom = mol.NewAtom()
        atom.SetAtomicNum(t)
        atom.SetVector(x,y,z)
        atoms.append(atom)
    return mol, atoms


def get_ring_sys(mol):
    all_rings = Chem.GetSymmSSSR(mol)
    if len(all_rings) == 0:
        ring_sys_list = []
    else:
        ring_sys_list = [all_rings[0]]
        for ring in all_rings[1:]:
            form_prev = False
            for prev_ring in ring_sys_list:
                if set(ring).intersection(set(prev_ring)):
                    prev_ring.extend(ring)
                    form_prev = True
                    break
            if not form_prev:
                ring_sys_list.append(ring)
    ring_sys_list = [list(set(x)) for x in ring_sys_list]
    return ring_sys_list


def get_all_subsets(ring_list):
    all_sub_list = []
    for n_sub in range(len(ring_list)+1):
        all_sub_list.extend(itertools.combinations(ring_list, n_sub))
    return all_sub_list


def fix_valence(mol):
    mol = copy.deepcopy(mol)
    fixed = False
    cnt_loop = 0
    while True:
        try:
            Chem.SanitizeMol(mol)
            fixed = True
            break
        except Chem.rdchem.AtomValenceException as e:
            err = e
        except Exception as e:
            return mol, False # from HERE: rerun sample
        cnt_loop += 1
        if cnt_loop > 100:
            break
        N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
        index = N4_valence.findall(err.args[0])
        if len(index) > 0:
            mol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
    return mol, fixed


def fix_aromatic(mol, strict=False):
    mol_orig = mol
    atomatic_list = [a.GetIdx() for a in mol.GetAromaticAtoms()]
    N_ring_list = []
    S_ring_list = []
    for ring_sys in get_ring_sys(mol):
        if set(ring_sys).intersection(set(atomatic_list)):
            idx_N = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'N']
            if len(idx_N) > 0:
                idx_N.append(-1) # -1 for not add to this loop
                N_ring_list.append(idx_N)
            idx_S = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'S']
            if len(idx_S) > 0:
                idx_S.append(-1) # -1 for not add to this loop
                S_ring_list.append(idx_S)
    # enumerate S
    fixed = False
    if strict:
        S_ring_list = [s for ring in S_ring_list for s in ring if s != -1]
        permutation = get_all_subsets(S_ring_list)
    else:
        permutation = list(itertools.product(*S_ring_list))
    for perm in permutation:
        mol = copy.deepcopy(mol_orig)
        perm = [x for x in perm if x != -1]
        for idx in perm:
            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
        try:
            if strict:
                mol, fixed = fix_valence(mol)
            Chem.SanitizeMol(mol)
            fixed = True
            break
        except:
            continue
    # enumerate N
    if not fixed:
        if strict:
            N_ring_list = [s for ring in N_ring_list for s in ring if s != -1]
            permutation = get_all_subsets(N_ring_list)
        else:
            permutation = list(itertools.product(*N_ring_list))
        for perm in permutation:  # each ring select one atom
            perm = [x for x in perm if x != -1]
            # print(perm)
            actions = itertools.product([0, 1], repeat=len(perm))
            for action in actions: # add H or charge
                mol = copy.deepcopy(mol_orig)
                for idx, act_atom in zip(perm, action):
                    if act_atom == 0:
                        mol.GetAtomWithIdx(idx).SetNumExplicitHs(1)
                    else:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                try:
                    if strict:
                        mol, fixed = fix_valence(mol)
                    Chem.SanitizeMol(mol)
                    fixed = True
                    break
                except:
                    continue
            if fixed:
                break
    return mol, fixed




def calc_valence(rdatom):
    '''Can call GetExplicitValence before sanitize, but need to
    know this to fix up the molecule to prevent sanitization failures'''
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt


def convert_ob_mol_to_rd_mol(ob_mol,struct=None):
    '''Convert OBMol to RDKit mol, fixing up issues'''
    ob_mol.DeleteHydrogens()
    n_atoms = ob_mol.NumAtoms()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
        #TODO copy format charge
        if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
            #don't commit to being aromatic unless rdkit will be okay with the ring status
            #(this can happen if the atoms aren't fit well enough)
            rd_atom.SetIsAromatic(True)
        i = rd_mol.AddAtom(rd_atom)
        ob_coords = ob_atom.GetVector()
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):
        i = ob_bond.GetBeginAtomIdx()-1
        j = ob_bond.GetEndAtomIdx()-1
        bond_order = ob_bond.GetBondOrder()
        if bond_order == 1:
            rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
        elif bond_order == 2:
            rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
        elif bond_order == 3:
            rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
        else:
            raise Exception('unknown bond order {}'.format(bond_order))

        if ob_bond.IsAromatic():
            bond = rd_mol.GetBondBetweenAtoms (i,j)
            bond.SetIsAromatic(True)

    rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)

    pt = Chem.GetPeriodicTable()
    #if double/triple bonds are connected to hypervalent atoms, decrement the order

    positions = rd_mol.GetConformer().GetPositions()
    nonsingles = []
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(positions[i]-positions[j])
            nonsingles.append((dist,bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])

    for (d,bond) in nonsingles:
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
           calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
            btype = Chem.BondType.SINGLE
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                btype = Chem.BondType.DOUBLE
            bond.SetBondType(btype)

    for atom in rd_mol.GetAtoms():
        #set nitrogens with 4 neighbors to have a charge
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)

    rd_mol = Chem.AddHs(rd_mol,addCoords=True)

    positions = rd_mol.GetConformer().GetPositions()
    center = np.mean(positions[np.all(np.isfinite(positions),axis=1)],axis=0)
    for atom in rd_mol.GetAtoms():
        i = atom.GetIdx()
        pos = positions[i]
        if not np.all(np.isfinite(pos)):
            #hydrogens on C fragment get set to nan (shouldn't, but they do)
            rd_mol.GetConformer().SetAtomPosition(i,center)

    try:
        Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
    except:
        raise MolReconsError()
    # try:
    #     Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
    # except: # mtr22 - don't assume mols will pass this
    #     pass
    #     # dkoes - but we want to make failures as rare as possible and should debug them
    #     m = pybel.Molecule(ob_mol)
    #     i = np.random.randint(1000000)
    #     outname = 'bad%d.sdf'%i
    #     print("WRITING",outname)
    #     m.write('sdf',outname,overwrite=True)
    #     pickle.dump(struct,open('bad%d.pkl'%i,'wb'))

    #but at some point stop trying to enforce our aromaticity -
    #openbabel and rdkit have different aromaticity models so they
    #won't always agree.  Remove any aromatic bonds to non-aromatic atoms
    for bond in rd_mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)

    return rd_mol

UPGRADE_BOND_ORDER = {Chem.BondType.SINGLE:Chem.BondType.DOUBLE, Chem.BondType.DOUBLE:Chem.BondType.TRIPLE}


def postprocess_rd_mol_1(rdmol):

    rdmol = Chem.RemoveHs(rdmol)

    # Construct bond nbh list
    nbh_list = {}
    for bond in rdmol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() 
        if begin not in nbh_list: nbh_list[begin] = [end]
        else: nbh_list[begin].append(end)
            
        if end not in nbh_list: nbh_list[end] = [begin]
        else: nbh_list[end].append(begin)

    # Fix missing bond-order
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            for j in nbh_list[idx]:
                if j <= idx: continue
                nb_atom = rdmol.GetAtomWithIdx(j)
                nb_radical = nb_atom.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rdmol.GetBondBetweenAtoms(idx, j)
                    bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                    nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                    num_radical -= 1
            atom.SetNumRadicalElectrons(num_radical)

        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            atom.SetNumRadicalElectrons(0)
            num_hs = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(num_hs + num_radical)
            
    return rdmol


def postprocess_rd_mol_2(rdmol):
    rdmol_edit = Chem.RWMol(rdmol)

    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]
    for i, ring_a in enumerate(rings):
        if len(ring_a) == 3:
            non_carbon = []
            atom_by_symb = {}
            for atom_idx in ring_a:
                symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != 'C':
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)
            if len(non_carbon) == 2:
                rdmol_edit.RemoveBond(*non_carbon)
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                rdmol_edit.RemoveBond(*atom_by_symb['O'])
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                )
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                )
    rdmol = rdmol_edit.GetMol()

    for atom in rdmol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return rdmol
    

def reconstruct_from_generated(mol_info):
    atomic_nums = mol_info['element']
    xyz = mol_info['atom_pos'].tolist()

    mol, atoms = make_obmol(xyz, atomic_nums)
    fixup(atoms, mol, )

    connect_the_dots(mol, atoms, 2)
    fixup(atoms, mol, )
    mol.EndModify()

    fixup(atoms, mol, )

    mol.AddPolarHydrogens()
    mol.PerceiveBondOrders()
    fixup(atoms, mol, )

    for (i,a) in enumerate(atoms):
        ob.OBAtomAssignTypicalImplicitHydrogens(a)
    fixup(atoms, mol, )

    mol.AddHydrogens()
    fixup(atoms, mol, )

    #make rings all aromatic if majority of carbons are aromatic
    for ring in ob.OBMolRingIter(mol):
        if 5 <= ring.Size() <= 6:
            carbon_cnt = 0
            aromatic_ccnt = 0
            for ai in ring._path:
                a = mol.GetAtom(ai)
                if a.GetAtomicNum() == 6:
                    carbon_cnt += 1
                    if a.IsAromatic():
                        aromatic_ccnt += 1
            if aromatic_ccnt >= carbon_cnt/2 and aromatic_ccnt != ring.Size():
                #set all ring atoms to be aromatic
                for ai in ring._path:
                    a = mol.GetAtom(ai)
                    a.SetAromatic(True)

    #bonds must be marked aromatic for smiles to match
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.IsAromatic() and a2.IsAromatic():
            bond.SetAromatic(True)
            
    mol.PerceiveBondOrders()

    rd_mol = convert_ob_mol_to_rd_mol(mol)

    # Post-processing
    rd_mol = postprocess_rd_mol_1(rd_mol)
    rd_mol = postprocess_rd_mol_2(rd_mol)

    return rd_mol


def reconstruct_from_generated_with_edges(mol_info, add_edge='predicted', check_validity=True):
    atomic_nums = mol_info['element']
    xyz = mol_info['atom_pos'].tolist()
    if add_edge == 'predicted':
        if 'bond_index' in mol_info and mol_info['bond_index'] is not None:
            bond_index = mol_info['bond_index'].tolist()
            bond_type = mol_info['bond_type'].tolist()
        else:
            raise ValueError(f'No bond information is provided while add_edge_mode is set to `{add_edge}`')
    elif add_edge == 'openbabel':
        try:
            return reconstruct_from_generated(mol_info)
        except:
            raise MolReconsError('Openbabel failed to reconstruct the molecule')
    elif add_edge == 'distance':
        bond_index, bond_type = predict_bonds(atomic_nums, np.array(xyz))
    else:
        raise ValueError(f'Invalid add_edge_mode: {add_edge}')

    n_atoms = len(atomic_nums)

    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    # add atoms and coordinates
    for i, atom in enumerate(atomic_nums):
        rd_atom = Chem.Atom(atom)
        rd_mol.AddAtom(rd_atom)
        rd_coords = Geometry.Point3D(*xyz[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)

    # add bonds
    for i, type_this in enumerate(bond_type):
        node_i, node_j = bond_index[0][i], bond_index[1][i]
        if node_i < node_j:
            if type_this == 1:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
            elif type_this == 2:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
            elif type_this == 3:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
            elif type_this == 4:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
            else:
                raise Exception('unknown bond order {}'.format(type_this))

    mol = rd_mol.GetMol()
    if check_validity:
        logger = RDLogger.logger()
        logger.setLevel(RDLogger.CRITICAL)
        try:
            Chem.SanitizeMol(mol)
            fixed = True
        except Exception as e:
            fixed = False
        
        if not fixed:
            try:
                Chem.Kekulize(copy.deepcopy(mol))
            except Chem.rdchem.KekulizeException as e:
                err = e
                if 'Unkekulized' in err.args[0]:
                    mol, fixed = fix_aromatic(mol)

        # valence error for N 
        if not fixed:
            mol, fixed = fix_valence(mol)
            
        if not fixed:
            mol, fixed = fix_aromatic(mol, True)
            
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise MolReconsError()
    return mol
    

    