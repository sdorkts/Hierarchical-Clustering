import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in scalar divide.*")

warnings.filterwarnings("ignore")

class FowlkesMallows:
    @staticmethod
    def fm_index(A1_clusters, A2_clusters, assume_sorted_vectors=False, warn=True):
        if not assume_sorted_vectors:
            sort_indices = np.argsort(A1_clusters)
            A1_clusters = np.array(A1_clusters)[sort_indices]
            A2_clusters = np.array(A2_clusters)[sort_indices]

        if np.any(pd.isnull(A1_clusters)) or np.any(pd.isnull(A2_clusters)):
            if warn:
                print("Warning: The clusterings have some NA's in them - returned None.")
            return {"FM": None, "E_FM": None, "V_FM": None}

        M = pd.crosstab(A1_clusters, A2_clusters).to_numpy()
        n = len(A1_clusters)
        
        Tk = np.sum(M**2) - n
        m_i_dot = np.sum(M, axis=1)
        m_dot_j = np.sum(M, axis=0)
        
        Pk = np.sum(m_i_dot**2) - n
        Qk = np.sum(m_dot_j**2) - n

        FM = Tk / np.sqrt(Pk * Qk) if Pk * Qk > 0 else np.nan  # Avoid division by zero
        FM = 0 if np.isnan(FM) else FM

        E_FM = np.sqrt(Pk * Qk) / (n * (n - 1))
        
        Pk2 = np.sum(m_i_dot * (m_i_dot - 1) * (m_i_dot - 2))
        Qk2 = np.sum(m_dot_j * (m_dot_j - 1) * (m_dot_j - 2))
        
        V_FM = (2 / (n * (n - 1))
                + (4 * Pk2 * Qk2) / ((n * (n - 1) * (n - 2)) * Pk * Qk)
                + ((Pk - 2 - 4 * Pk2 / Pk) * (Qk - 2 - 4 * Qk2 / Qk)) /
                (n * (n - 1) * (n - 2) * (n - 3))
                - Pk * Qk / (n**2 * (n - 1)**2))
        
        return {"FM": FM, "E_FM": E_FM, "V_FM": V_FM}

    @staticmethod
    def Bk(precomputed_clusters1, precomputed_clusters2, k_range=None, warn=True):
        if k_range is None:
            k_range = set(precomputed_clusters1.keys()) & set(precomputed_clusters2.keys())
        
        the_Bks = {}
        for k in k_range:
            if k in precomputed_clusters1 and k in precomputed_clusters2:
                clusters1 = precomputed_clusters1[k]
                clusters2 = precomputed_clusters2[k]
                fm_result = FowlkesMallows.fm_index(clusters1, clusters2)
                the_Bks[k] = fm_result
            else:
                if warn:
                    print(f"Warning: Missing cluster data for level {k}")
        
        return the_Bks

    
    @staticmethod
    def fm_is_sig(fm):
        threshold = fm['E_FM'] + 1.645 * np.sqrt(fm['V_FM'])
        print(f"FM: {fm['FM']}")
        print(f"Significance Threshold: {threshold}")
        print("This is Significant!" if fm['FM'] > threshold else "This is NOT Significant")

    @staticmethod
    def plot_fm_results(results_dict, important_cluster=None):
        clusters = list(results_dict.keys())
        fm_values = [results_dict[cluster]['FM'] for cluster in clusters]
        
        significance = []
        for cluster in clusters:
            fm = results_dict[cluster]
            threshold = fm['E_FM'] + 1.645 * np.sqrt(fm['V_FM'])
            significance.append(fm['FM'] > threshold)
        
        plt.figure(figsize=(8, 4))
        for i in range(1, len(clusters)):
            color = 'green' if significance[i] else 'red'
            plt.plot(clusters[i-1:i+1], fm_values[i-1:i+1], color=color, linewidth=2)
        
        plt.xlabel('Clusters')
        plt.ylabel('Bk (FM values)')
        plt.title('Bk vs Clusters with Significance')
        plt.axhline(y=0, color='black', linestyle='--')
        if important_cluster:
            plt.axvline(x=important_cluster, color='blue', linestyle='--', linewidth=2, label='Standard Clusters')
        plt.xticks(rotation=45)
        plt.scatter([], [], color='green', label='Significant')
        plt.scatter([], [], color='red', label='Not Significant')
        plt.legend()
        plt.tight_layout()
        plt.show()
