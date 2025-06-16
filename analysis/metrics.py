""" Metrics. """
import mdtraj as md
import numpy as np
from openfold.np import residue_constants
from tmtools import tm_align
from data import utils as du


def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2 

def calc_mdtraj_metrics(pdb_path):
    try:
        traj = md.load(pdb_path)
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_coil_percent = np.mean(pdb_ss == 'C')
        pdb_helix_percent = np.mean(pdb_ss == 'H')
        pdb_strand_percent = np.mean(pdb_ss == 'E')
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent 
        pdb_rg = md.compute_rg(traj)[0]
    except IndexError as e:
        print('Error in calc_mdtraj_metrics: {}'.format(e))
        pdb_ss_percent = 0.0
        pdb_coil_percent = 0.0
        pdb_helix_percent = 0.0
        pdb_strand_percent = 0.0
        pdb_rg = 0.0
    return {
        'non_coil_percent': pdb_ss_percent,
        'coil_percent': pdb_coil_percent,
        'helix_percent': pdb_helix_percent,
        'strand_percent': pdb_strand_percent,
        'radius_of_gyration': pdb_rg,
    }

def calc_ca_ca_metrics(ca_pos, bond_tol=0.1, clash_tol=1.0):
    ca_bond_dists = np.linalg.norm( # Computes the Euclidean distances between each Cα and the previous Cα using np.roll to shift the array for pairwise distance calculation.
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca)) # average deviation from the standard distances
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + bond_tol))
    
    ca_ca_dists2d = np.linalg.norm( # pariwise distance
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
    clashes = inter_dists < clash_tol
    return {
        'ca_ca_deviation': ca_ca_dev, # Mean deviation from the standard Cα–Cα bond length.
        'ca_ca_valid_percent': ca_ca_valid, # The percentage of Cα–Cα bonds within the defined tolerance range.
        'num_ca_ca_clashes': np.sum(clashes), # The count of close Cα–Cα contacts that are considered clashes.
    }

def calc_aligned_rmsd(pos_1, pos_2):
    aligned_pos_1 = du.rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))

import numpy as np
from scipy.special import rel_entr  # This computes the relative entropy, which is used for KL divergence

# Function to compute KL divergence
def kl_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    
    # Add small constant to avoid zero probabilities (for KL divergence stability)
    epsilon = 1e-10
    p = np.clip(p, epsilon, None)  # Avoid log(0) issues
    q = np.clip(q, epsilon, None)  # Avoid log(0) issues
    
    # Compute KL divergence
    kl_div = np.sum(rel_entr(p, q))  # KL divergence P || Q
    return kl_div

# Function to compute Jensen-Shannon Divergence (JSD) between 1D probability distributions #TODO: Compare to earth Movers distance
def js_divergence(a, b, nbins=20):
    """
    ARGS:
    a: 1 dimensional np.array containing samples from distribution A
    b: 1 dimensional np.array containing samples from distribution B
    nbins: number of bins to use for histogram (TODO: Experiment with different values here)
    """

    xmin = np.min(np.concatenate([a, b]))
    xmax = np.max(np.concatenate([a, b]))
    #Create bins between min and max
    bins = np.linspace(xmin, xmax, nbins+1)
    #Create histograms with the same bins for both distributions
    p, q = (np.histogram(a, bins=bins)[0],
            np.histogram(b, bins=bins)[0])
    # Normalize histograms to get probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Create the mixture distribution
    m = 0.5 * (p + q)
    
    # Compute KL divergence from p and q to m
    kl_pm = kl_divergence(p, m)  # KL divergence P || M
    kl_qm = kl_divergence(q, m)  # KL divergence Q || M
    
    # Jensen-Shannon Divergence is the average of the two KL divergences
    jsd = 0.5 * (kl_pm + kl_qm)
    
    return jsd