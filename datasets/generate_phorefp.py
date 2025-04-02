import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, RemoveHs

DEBUG = False
periodic_table = GetPeriodicTable()
# NUM_PHORETYPE = 11
NUM_PHORETYPE = 13
# PHORETYPES = ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV', 'CR', 'XB', 'EX']
PHORETYPES = ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV1', 'CV2', 'CV3', 'CV4', 'XB', 'EX']
PHORE_SMARTS = {
    'MB': {
        '*-P(-O)(-O)=O': [2, 3, 4],
        '*-S(-O)=O': [2, 3],
        '*-S(=O)(-O)=O': [2, 3, 4],
        '*-S(-*)=O': [3],
        '*-C(-O)=O': [2, 3],
        '[O^3]': [0],
        '*-C(-C(-F)(-F)-F)=O': [6],
        '[OH1]-P(-*)(-*)=O': [0, 4],
        '*-C(-N-*)=O': [4],
        '*-[CH1]=O': [2],
        '*-N(-*)-N=O': [4],
        '*-C(-S-*)=O': [4],
        'O=C(-C-O-*)-C-[OH1]': [0],
        '*-C(-S-*)=O': [4],
        '*-C(-C(-[OH1])=C)=O': [5],
        '[S^3D2]': [0],
        '*=N-C=S': [3],
        'S=C(-N-C(-*)=O)-N-C(-*)=O': [0],
        '[#7^2,#7^3;!$([n;H0;X3]);!+;!+2;!+3]': [0],
        '[C,#1]-[Se&H1]': [1],
        'C1:C:C:C:S:1': [4],
        'O2:C:C:C:C:2': [0],
        'a[O,NH2,NH1,SX2,SeH1]': [1],
        '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]' : [0]
    },

    'NE': {
        '[CX3,SX3,PD3](=[O,S])[O;H0&-1,OH1]': [1, 2],
        '[PX4](=[O,S])([O;H0&-1,OH1])[O;H0&-1,OH1]': [1, 2, 3],
        '[PX4](=[O,S])([O;H0&-1,OH1])[O][*;!H]': [1, 2],
        '[SX4](=[O,S])(=[O,S])([O;H0&-1,OH1])': [1, 2, 3]
    },

    'PO': {
        '[+;!$([N+]-[O-])]': [0],
        'N-C(-N)=N': [1]
    },

    'HD': {
        '[#7,#8,#16;+0,+1,+2;!H0]': [0]
    },

    'HA': {'[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]' : [0]
        # '[S;$(S=C);-0,-1,-2,-3]': [0],
    },
    # 'HA': {
    #     '[#7;!$([#7]~C=[N,O,S]);!$([#7]~S=O);!$([n;H0;X3]);!$([N;H1;X3]);-0,-1,-2,-3]': [0],
    #     '[O,F;-0,-1,-2,-3]': [0],
    #     '[S^3;X2;H0;!$(S=O);-0,-1,-2,-3]': [0],
    #     # '[S;$(S=C);-0,-1,-2,-3]': [0],
    # },

    'CV':{
        '[N]#[C]-[C,#1]': [1],
        '[C,#1]-[C]1-[C](-[C,#1])-[O]-1': [1, 2],
        '[C]=[C]-[C](-[N&H1]-[C,#1])=[O]': [0],
        '[S&H1]-[C,#1]': [0],
        '[C,#1]-[C]1-[C](-[C,#1])-[N]-1': [1, 2],
        '[C]=[C]-[S](=[O])(-[C,#1])=[O]': [0],
        '[F,Cl,Br,I]-[C]-[C,#1]': [1],
        '[C,#1]-[C](-[F,Cl,Br,I])-[C](-[C,N,O]-[C,#1])=[O]': [1],
        '[O]=[C](-[N]-[C,#1])-[C]#[C]': [5],
        '[C,#1]-[S](-[C,#1])=[O]': [1],
        '[C,#1]-[Se&H1]': [1],
        '[O]=[C](-[O]-[C,#1])-[C]#[C]': [5],
        '[S]=[C]=[N]-[C,#1]': [1],
        '[C,#1]-[S]-[S]-[C,#1]': [1, 2],
        '[C,#1]-[N,O]-[C](-[N,O]-[C,#1])=[O]': [2],
        '[C,#1]-[C](-[C](-[N]-[C,#1])=[O])=[O]': [1],
        '[C,#1]-[B](-[O&H1])-[O&H1]': [1],
        '[C,#1]-[C&H1]=[O]': [1],
        '[C,#1]-[S](-[F])(=[O])=[O]': [1],
        '[C,#1]-[S](-[C]=[C])(=[O])=[O]': [3],
        '[F,Cl,Br,I]-[C]-[C](-[C,#1])=[O]': [1]
    },
    'AR': {'[a]': [0]},
    'CR': {'[a]': [0], 
           '[+;!$([N+]-[O-])]': [0],
           'N-C(-N)=N': [1],
           },
    'XB': {'[#6]-[Cl,Br,I;X1]': [1]},
    'HY': {
            # refered to hydrophobic atom in 
            # /home/worker/software/anaconda3/envs/diffphore/lib/python3.9/site-packages/rdkit/Data/BaseFeatures.fdef
            '[c,s,S&H0&v2,Br,I,$([#6;+0;!$([#6;$([#6]~[#7,#8,#9])])])]': [0]
        },
    # CV-SH
    'CV1': {
        "C(-[F,Cl,Br,I])(-[!F;!Cl;!Br;!I])(-[!F;!Cl;!Br;!I])": [0],
        "C#N": [0],
        "C1-O-C-1": [0, 2],
        "[CX2]#C-C-N": [0],
        "C(=O)-O-C-C=O": [3],
        "C=C-C(=O)": [0],
        "S(=O)(=O)-C=C": [4],
        "C=C-[N+](-[O-])=O": [0],
        "N-C-[CH1,CH2]-C(=O)-C": [1],
        "[#6]1:[#7]:[#6]2:[#7H]:[#6]:[#7]:[#6]:2:[#6](-[#8]-[#6]-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2):[#7]:1": [10],
        "[#6]-C(=O)-[#1]": [1],
        "[#8]=[#6]1-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#6]=[#6]-[#6]-1=[#8]": [10],
        "N-C(=O)-C(=O)": [3],
        "O-C(=O)-N": [1],
        "N-C(=O)-S": [1],
        "[#8]=[#6](-[#7])-[#7]1:[#6]:[#7]:[#6]:[#6]:1": [1],
        "[#8]=[#6]1:[#8]:[#6](-[#8]):[#6]:[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:1:2": [1],
        "C1(=O)-O-C=N-N-1": [0],
        "N-C(=O)-n1:n:n:n:c:1": [1],
        "C1(=O)-[N,O]-C-C-1": [0],
        "[#6](=O)-,:[#8]": [0],
        "[NX3]-C(=S)-[NX3]": [1],
        "[#6]-[#16+](-[#6])-[#6]-[#6](-[#7])=[#8]": [1],
        "F-[SX4](=O)(=O)": [1],
        "[#6]-[SX2]-[H,CX4]": [1],
        "c1:c:c(-C(-F)(-F)(-F)):c:n:c:1-S(=O)(=O)-C": [9],
        "[#6]-[SX2]-[SX2]-[#6]": [1, 2],
        "C1=C-C-[NX3]-C-C-1": [0],
        "[#6]1:[#7H]:[#6](:[#6]2:[#6](:[#7]:1):[#7]:[#6]:[#7]:2)=[#8]": [0],
        "[NX2]=C=S": [0],
        "C=N-[OX2]-[#6]": [1],
        '[C,#1]-[C]1-[C](-[C,#1])-[N]-1': [1, 2],
        '[C,#1]-[C](-[F,Cl,Br,I])-[C](-[C,N,O]-[C,#1])=[O]': [1],
        '[O]=[C](-[N]-[C,#1])-[C]#[C]': [5],
        '[C,#1]-[S](-[C,#1])=[O]': [1],
        '[C,#1]-[Se&H1]': [1],
        '[O]=[C](-[O]-[C,#1])-[C]#[C]': [5],
        "[BX3](-O)(-O)": [0]
    },
    # CV-OH
    'CV2': {
        "C(-[F,Cl,Br,I])(-[!F;!Cl;!Br;!I])(-[!F;!Cl;!Br;!I])": [0],
        "C#N": [0],
        "C1-O-C-1": [0, 2],
        "C=C-C(=O)": [0],
        "[#6]-C(=O)-[#1]": [1],
        "O-C(=O)-N": [1],
        "[#8]=[#6]1-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#6]=[#6]-[#6]-1=[#8]": [10],
        "N-C(=O)-C(=O)": [3],
        "O-C(=O)-N": [1],
        "N-C(=O)-S": [1],
        "[#8]=[#6](-[#7])-[#7]1:[#6]:[#7]:[#6]:[#6]:1": [1],
        "[#8]=[#6]1:[#8]:[#6](-[#8]):[#6]:[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:1:2": [1],
        "C1(=O)-O-C=N-N-1": [0],
        "N-C(=O)-n1:n:n:n:c:1": [1],
        "C1(=O)-[N,O]-C-C-1": [0],
        "[#6](=O)-,:[#8H0]": [0],
        "[#6]-[#16+](-[#6])-[#6]-[#6](-[#7])=[#8]": [1],
        "F-[SX4](=O)(=O)": [1],
        "[NX2]=C=S": [0],
        "[CX3]1-N=N-1": [0],
        '[C,#1]-[N,O]-[C](-[N,O]-[C,#1])=[O]': [2],
        '[C,#1]-[C](-[C](-[N]-[C,#1])=[O])=[O]': [1],
        '[C,#1]-[B](-[O&H1])-[O&H1]': [1],
        "[BX3](-O)(-O)": [0],
        '[C,#1]-[C&H1]=[O]': [1]

    },
    # CV-NH2
    'CV3': {
        "C(-[F,Cl,Br,I])(-[!F;!Cl;!Br;!I])(-[!F;!Cl;!Br;!I])": [0],
        "C#N": [0],
        "C1-O-C-1": [0, 2],
        "C=C-C(=O)": [0],
        "S(=O)(=O)-C=C": [4],
        "[#6]-C(=O)-[#1]": [1],
        "[#8]=[#6]1-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#6]=[#6]-[#6]-1=[#8]": [10],
        "N-C(=O)-C(=O)": [3],
        "O-C(=O)-N": [1],
        "N-C(=O)-S": [1],
        "[#8]=[#6](-[#7])-[#7]1:[#6]:[#7]:[#6]:[#6]:1": [1],
        "[#8]=[#6]1:[#8]:[#6](-[#8]):[#6]:[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:1:2": [1],
        "C1(=O)-O-C=N-N-1": [0],
        "N-C(=O)-n1:n:n:n:c:1": [1],
        "C1(=O)-[N,O]-C-C-1": [0],
        "[#6](=O)-,:[#8]": [0],
        "[#6]-[#16+](-[#6])-[#6]-[#6](-[#7])=[#8]": [1],
        "F-[SX4](=O)(=O)": [1],
        "[NX2]=C=S": [0],
        "[OX2]-[PX4](=O)(-[F,Cl,Br,I])": [0],
        "[BX3](-O)(-O)": [0],
        "[NX2]=[N+]=[N-]": [0]
    },
    # CV-COOH
    'CV4': {
        "C(-[F,Cl,Br,I])(-[!F;!Cl;!Br;!I])(-[!F;!Cl;!Br;!I])": [0],
        "C1-O-C-1": [0, 2],
        '[F,Cl,Br,I]-[C]-[C](-[C,#1])=[O]': [1]
    }
}


def generate_ligand_phore_feat(mol, remove_hs=True):
    mol = RemoveHs(mol) if remove_hs else mol
    # rdPartialCharges.ComputeGasteigerCharges(mol)
    coords = mol.GetConformer().GetPositions()
    lig_phorefps = [] # [N, NUM_PHORETYPE]

    mol = analyze_phorefp(mol, remove_hs=remove_hs)

    for atom in mol.GetAtoms():
        phorefp = fetch_phorefeature(atom, coords)
        lig_phorefps.append(phorefp)

    return lig_phorefps


def analyze_phorefp(mol, remove_hs=True):
    _mol = Chem.AddHs(mol)
    hy_check(_mol)
    ha_check(_mol)
    if remove_hs: 
        _mol = Chem.RemoveHs(_mol)
    [phore_check(_mol, phoretype) for phoretype in PHORETYPES if phoretype not in ['HY', 'HA']]
    return _mol


def fetch_phorefeature(atom, coords):
    phorefp = check_atom_phoretype(atom)
    return phorefp


def check_atom_phoretype(atom):
    phorefp = [0] * NUM_PHORETYPE
    prop_dict = atom.GetPropsAsDict()
    for i in range(NUM_PHORETYPE):
        phorefp[i] = 1 if PHORETYPES[i] in prop_dict and prop_dict[PHORETYPES[i]] == True else 0
    return phorefp


def phore_check(mol, phoretype='MB', setTrue=True, debug=DEBUG):
    idxes = []
    if phoretype in PHORE_SMARTS:
        for smarts, loc in PHORE_SMARTS[phoretype].items():
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            for match in matches:
                for l in loc:
                    idxes.append(match[l])
                    # idxes.append(l)
                    if setTrue:
                        at = mol.GetAtomWithIdx(match[l])
                        at.SetBoolProp(phoretype, True)
                        if debug:
                            print(at.GetSymbol()+str(at.GetIdx()), f"Set as {phoretype}")
    # print(idxes)
    return list(set(idxes))


def ha_check(mol, debug=True):
    idxes = phore_check(mol, 'HA', setTrue=True)


def hy_check(mol, follow_ancphore=False):
    if follow_ancphore:
        labelLipoAtoms(mol)
        atomSet = set([])
        for at in mol.GetAtoms():
            at.SetBoolProp('HY', False)
            atomSet.add(at.GetIdx())
            if at.GetAtomicNum() != 1:
                t = at.GetDoubleProp('pcharge')
                if float_eq(t, 0):
                    at.SetDoubleProp('pcharge', calAccSurf(at, 'HY') * t)
        
        # Rings smaller than 7 atoms
        for ring in Chem.GetSSSR(mol):
            if len(ring) < 7:
                lipoSum = 0
                for at_idx in ring:
                    lipoSum += mol.GetAtomWithIdx(at_idx).GetDoubleProp('pcharge')
                    if at_idx in atomSet:
                        atomSet.remove(at_idx)
                if lipoSum > 9.87:
                    for at_idx in ring:
                        mol.GetAtomWithIdx(at_idx).SetBoolProp('HY', True)

        # Atoms with three or more bonds
        for at_idx in atomSet:
            at = mol.GetAtomWithIdx(at_idx)
            collected_idx = [at_idx]
            if at.GetTotalNumHs() > 2:
                lipoSum = at.GetDoubleProp('pcharge')
                for bond in at.GetBonds():
                    neib = bond.GetOtherAtom(at)
                    if neib.GetTotalNumHs() == 1 and at.GetAtomicNum() != 1:
                        lipoSum += neib.GetDoubleProp('pcharge')
                        collected_idx.append(neib.GetIdx())
                if lipoSum > 9.87:
                    for at_idx in collected_idx:
                        mol.GetAtomWithIdx(at_idx).SetBoolProp('HY', True)
    else:
        phore_check(mol, 'HY', setTrue=True)


def calAccSurf(atom, mode='HA'):
    mol = atom.GetOwningMol()
    coords = mol.GetConformer().GetPositions()
    coord = coords[atom.GetIdx()]
    radius = periodic_table.GetRvdw(atom.GetAtomicNum())
    if mode == 'HA':
        radius = 1.8 
    elif mode == 'HY':
        radius = periodic_table.GetRvdw(atom.GetAtomicNum())


    arclength = 1.0 / np.sqrt(np.sqrt(3.0) * 2.0)
    dphi = arclength / radius
    nlayer = int(np.pi / dphi) + 1
    phi = 0.0
    sphere = []
    for i in range(nlayer):
        rsinphi = radius * np.sin(phi)
        z = radius * np.cos(phi)
        dtheta = 2 * np.pi if rsinphi == 0 else arclength / rsinphi
        tmpNbrPoints = int(2 * np.pi / dtheta)
        if tmpNbrPoints <= 0:
            tmpNbrPoints = 1
        dtheta = np.pi * 2.0 / tmpNbrPoints
        theta = 0 if i % 2 else np.pi
        for j in range(tmpNbrPoints):
            sphere.append(np.array([rsinphi*np.cos(theta)+coord[0], rsinphi*np.sin(theta)+coord[1], z+coord[2]]))
            theta += dtheta
            if theta > np.pi * 2:
                theta -= np.pi * 2
        phi += dphi
    
    aList = []
    if mode == 'HA':
        aList = [at for at in mol.GetAtoms()\
                    if np.sum(np.square(coords[at.GetIdx()] - coord)) \
                        <= np.square(3.0 + periodic_table.GetRvdw(at.GetAtomicNum()))\
                    and at.GetIdx() != atom.GetIdx()]
    elif mode == 'HY':
        aList = [at for at in mol.GetAtoms()\
                    if np.sum(np.square(coords[at.GetIdx()] - coord)) \
                        <= np.square(periodic_table.GetRvdw(atom.GetAtomicNum()) + \
                                     periodic_table.GetRvdw(at.GetAtomicNum()) + 2.8)\
                    and at.GetIdx() != atom.GetIdx()]
    
    delta = 1 if mode != 'HY' else 1.4 / radius

    nbrAccSurfPoints = 0
    prob_r = 1.2 if mode != 'HY' else 1.4
    isAccessible = True
    for s in sphere:
        p = s if mode != 'HY' else (s - coord) * delta + s
        for at in aList:
            distSq = np.sum(np.square(coords[at.GetIdx()] - p))
            r = periodic_table.GetRvdw(at.GetAtomicNum())
            sumSq = np.square(r+prob_r)
            if distSq <= sumSq:
                isAccessible = False
                break
        if isAccessible:
            nbrAccSurfPoints += 1
    if mode == 'HA':
        return float(nbrAccSurfPoints / len(sphere))
    elif mode == 'HY':
        return float(nbrAccSurfPoints / len(sphere) * 4 * np.pi * radius * radius)


def labelLipoAtoms(m):
    # pcharges = [1.0] * len(m.GetAtoms())
    for at in m.GetAtoms():
        at.SetDoubleProp("pcharge", 1.0)

    for at in m.GetAtoms():
        at_num = at.GetAtomicNum()
        if at_num == 1:
            at.SetDoubleProp("pcharge", 0.0)

        elif at_num == 7:
            at.SetDoubleProp("pcharge", 0.0)
            if not at.GetIsAromatic():
                labelLipoNeighbors(at, 0.25)
                if at.GetTotalNumHs() != 0:
                    for bond in at.GetBonds():
                        neib = bond.GetOtherAtom(at)
                        neib.SetDoubleProp('pcharge', 0.0)
                        labelLipoNeighbors(neib, 0.0)

        elif at_num == 8:
            at.SetDoubleProp("pcharge", 0.0)
            if not at.GetIsAromatic():
                labelLipoNeighbors(at, 0.25)
                for bond in at.GetBonds():
                    neib = bond.GetOtherAtom(at)
                    if neib.GetAtomicNum() == 1:
                        for bond1 in at.GetBonds():
                            nneib = bond1.GetOtherAtom(at)
                            nneib.SetDoubleProp('pcharge', 0.0)
                            labelLipoNeighbors(nneib, 0.0)
                    if bond.GetBondType().name == "DOUBLE":
                        neib.SetDoubleProp('pcharge', 0.0)
                        for bond1 in neib.GetBonds():
                            nneib = bond1.GetOtherAtom(neib)
                            if nneib.GetIdx() == at.GetIdx():
                                continue
                            nneib.SetDoubleProp('pcharge', 0.0)
                            labelLipoNeighbors(nneib, 0.6)

        elif at_num == 16:
            for bond in at.GetBonds():
                neib = bond.GetOtherAtom(at)
                if neib.GetAtomicNum() == 1:
                    at.SetDoubleProp('pcharge', 0.0)
                    labelLipoNeighbors(at, 0.0)
                if bond.GetBondType().name == "DOUBLE":
                    at.SetDoubleProp('pcharge', 0.0)
                    labelLipoNeighbors(at, 0.6)
            
            if at.GetTotalNumHs() > 2:
                at.SetDoubleProp('pcharge', 0.0)
                for bond in at.GetBonds():
                    neib = bond.GetOtherBonds(at)
                    neib.SetDoubleProp('pcharge', 0.0)
                    labelLipoNeighbors(at, 0.6)

        if at.GetFormalCharge() != 0:
            for bond in at.GetBonds():
                neib = bond.GetOtherAtom(at)
                neib.SetDoubleProp('pcharge', 0.0)
                labelLipoNeighbors(neib, 0.0)
    
    for at in m.GetAtoms():
        value = at.GetDoubleProp('pcharge')
        if (float_eq(value, 0.36) or value < 0.25) and not float_eq(value, 0.15):
            at.SetDoubleProp('pcharge', 0.0)


def float_eq(a, b, epsilon=1e-6):
    return abs(a - b) <= epsilon
            

def labelLipoNeighbors(atom, value):
    for bond in atom.GetBonds():
        neib = bond.GetOtherAtom(atom)
        neib.SetDoubleProp('pcharge', value * neib.GetDoubleProp('pcharge'))
