import os
import torch
from torch.nn import functional as F
from torch_geometric.data import Dataset, HeteroData


## Global Parameters
PHORETYPES = ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV', 'CR', 'XB', 'EX']
PHORETYPES1 = ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV1', 'CV2', 'CV3', 'CV4', 'XB', 'EX']


class PhoreData_New(Dataset):
    """
        Input: phore file.
        Args:
            file_list: [phore1, phore2, phore3, ...]
    """
    def __init__(self, file_list, center='phore', transform=None, data_name='zinc_300'):
        super(PhoreData_New, self).__init__(transform=transform)
        self.file_list = file_list
        self.center = center
        self.data_name = data_name

    def parse_phore_file(self, phore_file, data):
        if phore_file is not None and os.path.exists(phore_file):
            all_phore_type = PHORETYPES1 if self.data_name in ['zinc_300', 'pdbbind'] else PHORETYPES
            possible_phore_type = {phore_type: index for index, phore_type in enumerate(all_phore_type)}
            phore_type_list, alpha_list, pos_list, has_norm_list, norm_list = [], [], [], [], []
            with open(phore_file, 'r') as f:
                title = f.readline().strip()
                record = f.readline()
                while record:
                    record = record.strip()
                    if record != "$$$$":
                        try:
                            phore_type, alpha, weight, factor, x, y, z, \
                                has_norm, norm_x, norm_y, norm_z, label, anchor_weight = record.split("\t")
                            if phore_type == 'CR':
                                # print(f"Unsported phore type: {phore_type}")
                                record = f.readline()
                                continue
                            if phore_type == 'CV':
                                phore_type += label[0]
                            phore_type_list.append(possible_phore_type[phore_type])
                            alpha_list.append(float(alpha))
                            pos_list.append([float(x), float(y), float(z)])
                            has_norm_list.append(int(has_norm))
                            norm_list.append([float(norm_x), float(norm_y), float(norm_z)])
                        except Exception as e:
                            print(f"[E]: Failed to parse the line:\n {record} | Message: {e}")
                    else:
                        break
                    record = f.readline()
                    
            phore_type_t = F.one_hot(torch.tensor(phore_type_list, dtype=torch.long), num_classes=len(possible_phore_type)).float()
            exclusion_volume = F.one_hot(phore_type_t[:, -1:].long().squeeze(-1), 2).float()
            alpha_t = torch.tensor(alpha_list, dtype=torch.float).unsqueeze(-1)
            has_norm_t = F.one_hot(torch.tensor(has_norm_list, dtype=torch.long), 2).float()

            ## get unit norm
            norm_t = torch.tensor(norm_list, dtype=torch.float)
            norms = norm_t.norm(dim=-1, keepdim=True)
            no_zero = (norms != 0).squeeze(-1)
            unit_norm = torch.zeros_like(norm_t)
            unit_norm[no_zero] = norm_t[no_zero] / norms[no_zero]

            data['phore'].pos = torch.tensor(pos_list, dtype=torch.float)
            data['phore'].x = torch.cat((phore_type_t, alpha_t, has_norm_t, exclusion_volume), dim=-1)
            data['phore'].norm = unit_norm
            data['phore'].center_of_mass = torch.tensor(pos_list, dtype=torch.float).mean(dim=0)
            return data
        else:
            raise FileNotFoundError(f"The specified pharmacophore file (*.phore) is not found: `{phore_file}`")

    def get_empty_ligand_data(self, data):
        data['ligand'].atom_count = torch.empty([0,], dtype=torch.long)
        data['ligand'].pos = torch.empty([0, 3], dtype=torch.float)
        data['ligand'].center_of_mass = torch.empty([0,], dtype=torch.float)
        data['ligand', 'lig_bond', 'ligand'].edge_index = torch.empty([2, 0], dtype=torch.long)
        data['ligand', 'lig_bond', 'ligand'].edge_attr = torch.empty([0,], dtype=torch.long)
        data['ligand'].x = torch.empty([0, 12], dtype=torch.long)
        return data
    
    def move_to_center(self, center, data):
        if center == "phore":
            data["ligand"].pos -= data["phore"].center_of_mass
            data["phore"].pos -= data["phore"].center_of_mass
            data.center = data["phore"].center_of_mass
        elif center == "ligand":
            data["ligand"].pos -= data["ligand"].center_of_mass
            data["phore"].pos -= data["ligand"].center_of_mass
            data.center = data["ligand"].center_of_mass
        return data

    def len(self):
        return len(self.file_list)
    
    def get(self, idx):
        _data = HeteroData()
        phore_file = self.file_list[idx]
        _data.name = os.path.splitext(os.path.basename(phore_file))[0]
        data = self.parse_phore_file(phore_file, _data)
        data = self.get_empty_ligand_data(data)
        data = self.move_to_center(self.center, data)
        return data


class PhoreData(Dataset):
    """
        Input: phore file.
        Args:
            file_list: [phore1, phore2, phore3, ...]
    """
    def __init__(self, file_list, center='phore', transform=None, data_name='zinc'):
        super(PhoreData, self).__init__(transform=transform)
        self.file_list = file_list
        self.center = center
        self.data_name = data_name

    def parse_phore_file(self, phore_file, data):
        if phore_file is not None and os.path.exists(phore_file):
            with open(phore_file, 'r') as f:
                all_phore_type = PHORETYPES1 if self.data_name in ['zinc_300', 'pdbbind'] else PHORETYPES
                possible_phore_type = {phore_type: index for index, phore_type in enumerate(all_phore_type)}
                possible_has_norm = torch.tensor([True, False], dtype=torch.bool)
                possible_EX_type = torch.tensor([True, False], dtype=torch.bool)
                phore_type_list, alpha_list, pos_list, has_norm_list, norm_list = [], [], [], [], []
                title = f.readline().strip()
                record = f.readline()
                while record:
                    record = record.strip()
                    if record != "$$$$":
                        try:
                            phore_type, alpha, weight, factor, x, y, z, \
                                has_norm, norm_x, norm_y, norm_z, label, anchor_weight = record.split("\t")
                            if phore_type == 'CR':
                                # print(f"Unsported phore type: {phore_type}")
                                record = f.readline()
                                continue
                            if phore_type == 'CV':
                                phore_type += label[0]
                            phore_type = possible_phore_type[phore_type]
                            alpha = float(alpha)
                            pos = [float(x), float(y), float(z)]
                            has_norm = bool(int(has_norm))
                            norm = [float(norm_x), float(norm_y), float(norm_z)]
                            phore_type_list.append(phore_type)
                            alpha_list.append(alpha)
                            pos_list.append(pos)
                            has_norm_list.append(has_norm)
                            norm_list.append(norm)
                        except:
                            print(f"[E]: Failed to parse the line:\n {record}")
                    else:
                        break
                    record = f.readline()
                    
            pos_tensor = torch.tensor(pos_list, dtype=torch.float32)
            has_norm_tensor = torch.tensor(has_norm_list, dtype=torch.bool).view(-1, 1) == possible_has_norm.view(1, -1)
            norm_tensor = torch.tensor(norm_list, dtype=torch.float32)
            center_of_mass = pos_tensor.mean(dim=0)

            # change the norm_tensor to unit_vector
            drection_tensor = norm_tensor - torch.where(norm_tensor == 0, torch.tensor(0.0), pos_tensor)
            drection_magnitude = torch.sqrt((drection_tensor ** 2).sum(dim=1))
            nonzero_mask = drection_magnitude != 0
            drection_normalized_norm = torch.where(nonzero_mask.unsqueeze(1), 
                                            drection_tensor / drection_magnitude.unsqueeze(1), drection_tensor)
            
            type_tensor = F.one_hot(torch.tensor(phore_type_list))
            alpha_tensor = torch.tensor(alpha_list, dtype=torch.float32).unsqueeze(-1)
            exclusion_volume = type_tensor[:, -1]
            exclusion_volume = exclusion_volume.view(-1, 1) == possible_EX_type.view(1, -1)
            x = torch.cat([type_tensor, alpha_tensor, has_norm_tensor, exclusion_volume], dim=-1)
            
            data["phore"].pos = pos_tensor
            data['phore'].center_of_mass = center_of_mass
            data['phore'].x = x
            data["phore"].norm = drection_normalized_norm
            return data
        else:
            raise FileNotFoundError(f"The specified pharmacophore file (*.phore) is not found: `{phore_file}`")

    def get_empty_ligand_data(self, data):
        data['ligand'].atom_count = torch.empty([0,], dtype=torch.long)
        data['ligand'].pos = torch.empty([0, 3], dtype=torch.float)
        data['ligand'].center_of_mass = torch.empty([0,], dtype=torch.float)
        data['ligand', 'lig_bond', 'ligand'].edge_index = torch.empty([2, 0], dtype=torch.long)
        data['ligand', 'lig_bond', 'ligand'].edge_attr = torch.empty([0,], dtype=torch.long)
        data['ligand'].x = torch.empty([0, 12], dtype=torch.long)
        return data
    
    def move_to_center(self, center, data):
        if center == "phore":
            data["ligand"].pos -= data["phore"].center_of_mass
            data["phore"].pos -= data["phore"].center_of_mass
            data.center = data["phore"].center_of_mass
        elif center == "ligand":
            data["ligand"].pos -= data["ligand"].center_of_mass
            data["phore"].pos -= data["ligand"].center_of_mass
            data.center = data["ligand"].center_of_mass
        return data

    def len(self):
        return len(self.file_list)
    
    def get(self, idx):
        _data = HeteroData()
        phore_file = self.file_list[idx]
        _data.name = os.path.splitext(os.path.basename(phore_file))[0]
        data = self.parse_phore_file(phore_file, _data)
        data = self.get_empty_ligand_data(data)
        data = self.move_to_center(self.center, data)
        return data

