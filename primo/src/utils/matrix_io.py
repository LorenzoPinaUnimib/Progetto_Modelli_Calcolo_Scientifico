from scipy.io import mmread
from scipy.sparse import csr_matrix

def load_mtx(filepath):
    """Carica una matrice .mtx e la converte in formato CSR."""
    print(f"Caricamento matrice da {filepath}...")
    A = mmread(filepath)
    return csr_matrix(A)