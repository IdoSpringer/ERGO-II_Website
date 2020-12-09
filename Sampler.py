import pandas as pd


def read_input_file(datafile):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    data = pd.read_csv(datafile)
    for index in range(len(data)):
        sample = {}
        sample['tcra'] = data['TRA'][index]
        sample['tcrb'] = data['TRB'][index]
        sample['va'] = data['TRAV'][index]
        sample['ja'] = data['TRAJ'][index]
        sample['vb'] = data['TRBV'][index]
        sample['jb'] = data['TRBJ'][index]
        sample['t_cell_type'] = data['T-Cell-Type'][index]
        sample['peptide'] = data['Peptide'][index]
        sample['mhc'] = data['MHC'][index]
        # we do not use the sign
        sample['sign'] = 0
        if invalid(sample['tcrb']) or invalid(sample['peptide']):
            continue
        if invalid(sample['tcra']):
            sample['tcra'] = 'UNK'
        all_pairs.append(sample)
    return all_pairs, data

