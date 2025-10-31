"""
RNA Secondary Structure Predictor
Predicts RNA folding using Nussinov algorithm with minimum free energy calculation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import json
from datetime import datetime

class RNAStructurePredictor:
    """
    RNA Secondary Structure Predictor using Dynamic Programming
    """
    
    def __init__(self):
        # Base pairing energy parameters (kcal/mol) - simplified Turner rules
        self.base_pair_energy = {
            'GC': -3.0, 'CG': -3.0,
            'AU': -2.0, 'UA': -2.0,
            'GU': -1.5, 'UG': -1.5
        }
        
        # Stacking energy parameters
        self.stacking_energy = {
            'GC-GC': -3.4, 'CG-CG': -3.4,
            'AU-AU': -1.1, 'UA-UA': -1.1,
            'GC-CG': -2.4, 'CG-GC': -2.4,
            'AU-GC': -2.1, 'GC-AU': -2.1,
            'AU-CG': -2.1, 'CG-AU': -2.1,
            'UA-GC': -2.1, 'GC-UA': -2.1,
            'UA-CG': -2.1, 'CG-UA': -2.1,
            'GU-GC': -1.5, 'GC-GU': -1.5,
            'UG-GC': -1.5, 'GC-UG': -1.5,
        }
        
        self.min_loop_size = 3  # Minimum hairpin loop size
        
    def can_pair(self, base1: str, base2: str) -> bool:
        """Check if two bases can form a pair"""
        pair = base1 + base2
        return pair in self.base_pair_energy
    
    def get_pair_energy(self, base1: str, base2: str) -> float:
        """Get energy for a base pair"""
        pair = base1 + base2
        return self.base_pair_energy.get(pair, 0.0)
    
    def get_stacking_energy(self, i: int, j: int, sequence: str, pairs: np.ndarray) -> float:
        """Calculate stacking energy contribution"""
        if i + 1 < j - 1 and pairs[i + 1] == j - 1:
            inner = sequence[i + 1] + sequence[j - 1]
            outer = sequence[i] + sequence[j]
            stack_key = f"{outer}-{inner}"
            return self.stacking_energy.get(stack_key, -0.5)
        return 0.0
    
    def nussinov_algorithm(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Nussinov dynamic programming algorithm for RNA folding
        Returns: (dp_table, energy_table)
        """
        n = len(sequence)
        dp = np.zeros((n, n), dtype=int)
        energy_table = np.zeros((n, n), dtype=float)
        
        # Fill DP table
        for length in range(self.min_loop_size + 1, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Case 1: j unpaired
                if j > 0:
                    dp[i][j] = dp[i][j - 1]
                    energy_table[i][j] = energy_table[i][j - 1]
                
                # Case 2: j pairs with some k
                for k in range(i, j - self.min_loop_size):
                    if self.can_pair(sequence[k], sequence[j]):
                        left_pairs = dp[i][k - 1] if k > i else 0
                        middle_pairs = dp[k + 1][j - 1] if k + 1 < j else 0
                        total_pairs = left_pairs + 1 + middle_pairs
                        
                        if total_pairs > dp[i][j]:
                            dp[i][j] = total_pairs
                            
                            left_energy = energy_table[i][k - 1] if k > i else 0
                            middle_energy = energy_table[k + 1][j - 1] if k + 1 < j else 0
                            pair_energy = self.get_pair_energy(sequence[k], sequence[j])
                            
                            energy_table[i][j] = left_energy + middle_energy + pair_energy
        
        return dp, energy_table
    
    def traceback(self, dp: np.ndarray, sequence: str, i: int, j: int, pairs: np.ndarray):
        """Traceback to find the optimal structure"""
        if i >= j - self.min_loop_size:
            return
        
        # Check if j is unpaired
        if dp[i][j] == dp[i][j - 1]:
            self.traceback(dp, sequence, i, j - 1, pairs)
            return
        
        # Find pairing partner
        for k in range(i, j - self.min_loop_size):
            if self.can_pair(sequence[k], sequence[j]):
                left_pairs = dp[i][k - 1] if k > i else 0
                middle_pairs = dp[k + 1][j - 1] if k + 1 < j else 0
                
                if dp[i][j] == left_pairs + 1 + middle_pairs:
                    pairs[k] = j
                    pairs[j] = k
                    if k > i:
                        self.traceback(dp, sequence, i, k - 1, pairs)
                    if k + 1 < j:
                        self.traceback(dp, sequence, k + 1, j - 1, pairs)
                    return
    
    def predict_structure(self, sequence: str) -> Dict:
        """
        Predict RNA secondary structure
        Returns: Dictionary with structure, energy, and statistics
        """
        # Clean and validate sequence
        sequence = sequence.upper().replace('T', 'U')
        sequence = ''.join([b for b in sequence if b in 'AUGC'])
        
        if len(sequence) < 4:
            raise ValueError("Sequence must be at least 4 nucleotides long")
        
        n = len(sequence)
        
        # Run Nussinov algorithm
        dp, energy_table = self.nussinov_algorithm(sequence)
        
        # Traceback to find structure
        pairs = np.full(n, -1, dtype=int)
        self.traceback(dp, sequence, 0, n - 1, pairs)
        
        # Calculate total energy with stacking
        total_energy = 0.0
        for i in range(n):
            if pairs[i] > i:
                total_energy += self.get_pair_energy(sequence[i], sequence[pairs[i]])
                total_energy += self.get_stacking_energy(i, pairs[i], sequence, pairs)
        
        # Convert to dot-bracket notation
        dot_bracket = []
        for i in range(n):
            if pairs[i] == -1:
                dot_bracket.append('.')
            elif pairs[i] > i:
                dot_bracket.append('(')
            else:
                dot_bracket.append(')')
        
        structure = ''.join(dot_bracket)
        
        # Calculate statistics
        stats = self._calculate_statistics(sequence, structure, total_energy)
        
        return {
            'sequence': sequence,
            'structure': structure,
            'energy': total_energy,
            'pairs': pairs,
            'statistics': stats,
            'dp_table': dp,
            'energy_table': energy_table
        }
    
    def _calculate_statistics(self, sequence: str, structure: str, energy: float) -> Dict:
        """Calculate various statistics for the structure"""
        return {
            'length': len(sequence),
            'base_pairs': structure.count('('),
            'unpaired': structure.count('.'),
            'free_energy': round(energy, 2),
            'composition': {
                'A': sequence.count('A'),
                'U': sequence.count('U'),
                'G': sequence.count('G'),
                'C': sequence.count('C')
            },
            'gc_content': round((sequence.count('G') + sequence.count('C')) / len(sequence) * 100, 2)
        }
    
    def visualize_structure(self, result: Dict, save_path: str = None):
        """Visualize RNA secondary structure using matplotlib"""
        sequence = result['sequence']
        structure = result['structure']
        pairs = result['pairs']
        energy = result['energy']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'RNA Secondary Structure Analysis\nΔG = {energy:.2f} kcal/mol', 
                     fontsize=16, fontweight='bold')
        
        # 1. Arc Diagram
        ax1 = axes[0, 0]
        self._plot_arc_diagram(ax1, sequence, structure, pairs)
        
        # 2. Dot-Bracket Visualization
        ax2 = axes[0, 1]
        self._plot_dotbracket(ax2, sequence, structure)
        
        # 3. Energy Matrix Heatmap
        ax3 = axes[1, 0]
        self._plot_energy_heatmap(ax3, result['energy_table'])
        
        # 4. Statistics
        ax4 = axes[1, 1]
        self._plot_statistics(ax4, result['statistics'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def _plot_arc_diagram(self, ax, sequence, structure, pairs):
        """Plot arc diagram showing base pairs"""
        n = len(sequence)
        ax.set_xlim(-1, n)
        ax.set_ylim(-0.5, 3)
        
        # Color map for bases
        colors = {'A': '#FF6B6B', 'U': '#4ECDC4', 'G': '#FFE66D', 'C': '#95E1D3'}
        
        # Plot bases
        for i in range(n):
            color = colors.get(sequence[i], 'gray')
            ax.scatter(i, 0, s=200, c=color, edgecolors='black', linewidths=2, zorder=3)
            ax.text(i, -0.3, sequence[i], ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Plot arcs for base pairs
        for i in range(n):
            if pairs[i] > i:
                j = pairs[i]
                x = np.linspace(i, j, 100)
                height = (j - i) * 0.15
                y = height * np.sin(np.pi * (x - i) / (j - i))
                ax.plot(x, y, 'b-', linewidth=2, alpha=0.6)
        
        ax.set_title('Arc Diagram', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _plot_dotbracket(self, ax, sequence, structure):
        """Plot sequence and structure alignment"""
        ax.text(0.1, 0.7, 'Sequence:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.5, sequence, fontsize=10, fontfamily='monospace', transform=ax.transAxes)
        ax.text(0.1, 0.3, 'Structure:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.1, structure, fontsize=10, fontfamily='monospace', transform=ax.transAxes)
        ax.set_title('Dot-Bracket Notation', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _plot_energy_heatmap(self, ax, energy_table):
        """Plot energy matrix as heatmap"""
        sns.heatmap(energy_table, cmap='RdYlBu_r', center=0, ax=ax, 
                    cbar_kws={'label': 'Energy (kcal/mol)'})
        ax.set_title('Energy Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Position j')
        ax.set_ylabel('Position i')
    
    def _plot_statistics(self, ax, stats):
        """Plot statistics as text"""
        text = f"""
        Statistics:
        ───────────────────────────
        Length: {stats['length']} nt
        Base Pairs: {stats['base_pairs']}
        Unpaired: {stats['unpaired']}
        Free Energy: {stats['free_energy']} kcal/mol
        GC Content: {stats['gc_content']}%
        
        Base Composition:
        ───────────────────────────
        A: {stats['composition']['A']} ({stats['composition']['A']/stats['length']*100:.1f}%)
        U: {stats['composition']['U']} ({stats['composition']['U']/stats['length']*100:.1f}%)
        G: {stats['composition']['G']} ({stats['composition']['G']/stats['length']*100:.1f}%)
        C: {stats['composition']['C']} ({stats['composition']['C']/stats['length']*100:.1f}%)
        """
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace')
        ax.set_title('Structural Statistics', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def compare_structures(self, sequences: List[str], labels: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple RNA sequences
        Returns: DataFrame with comparison results
        """
        if labels is None:
            labels = [f"Sequence {i+1}" for i in range(len(sequences))]
        
        results = []
        for seq, label in zip(sequences, labels):
            try:
                result = self.predict_structure(seq)
                results.append({
                    'Label': label,
                    'Length': result['statistics']['length'],
                    'Base Pairs': result['statistics']['base_pairs'],
                    'Unpaired': result['statistics']['unpaired'],
                    'Free Energy (kcal/mol)': result['statistics']['free_energy'],
                    'GC Content (%)': result['statistics']['gc_content'],
                    'Structure': result['structure']
                })
            except Exception as e:
                print(f"Error processing {label}: {e}")
        
        return pd.DataFrame(results)
    
    def export_results(self, result: Dict, filename: str = "rna_structure_results.json"):
        """Export results to JSON file"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'sequence': result['sequence'],
            'structure': result['structure'],
            'energy': result['energy'],
            'statistics': result['statistics']
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {filename}")


def main():
    """Main function demonstrating the RNA Structure Predictor"""
    
    print("=" * 70)
    print("RNA SECONDARY STRUCTURE PREDICTOR".center(70))
    print("=" * 70)
    print()
    
    # Initialize predictor
    predictor = RNAStructurePredictor()
    
    # Example sequences
    examples = {
        "Simple Hairpin": "GGGAAACCC",
        "Stem Loop": "CGCGAAAGCG",
        "Complex Structure": "GGGCCCAUAGGGCCC",
        "tRNA-like": "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"
    }
    
    print("Available example sequences:")
    for i, (name, seq) in enumerate(examples.items(), 1):
        print(f"{i}. {name}: {seq[:30]}{'...' if len(seq) > 30 else ''}")
    print(f"{len(examples) + 1}. Enter custom sequence")
    print()
    
    choice = input(f"Select an option (1-{len(examples) + 1}): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(examples):
        seq_name = list(examples.keys())[int(choice) - 1]
        sequence = examples[seq_name]
        print(f"\nSelected: {seq_name}")
    elif choice == str(len(examples) + 1):
        sequence = input("Enter RNA sequence (AUGC): ").strip()
        seq_name = "Custom"
    else:
        print("Invalid choice. Using default hairpin sequence.")
        sequence = examples["Simple Hairpin"]
        seq_name = "Simple Hairpin"
    
    print("\nPredicting structure...")
    
    try:
        # Predict structure
        result = predictor.predict_structure(sequence)
        
        # Display results
        print("\n" + "=" * 70)
        print("RESULTS".center(70))
        print("=" * 70)
        print(f"\nSequence:  {result['sequence']}")
        print(f"Structure: {result['structure']}")
        print(f"\nFree Energy: {result['energy']:.2f} kcal/mol")
        print(f"Length: {result['statistics']['length']} nucleotides")
        print(f"Base Pairs: {result['statistics']['base_pairs']}")
        print(f"Unpaired Bases: {result['statistics']['unpaired']}")
        print(f"GC Content: {result['statistics']['gc_content']}%")
        print()
        
        # Visualize
        print("Generating visualization...")
        predictor.visualize_structure(result, save_path=f"rna_structure_{seq_name.replace(' ', '_')}.png")
        
        # Export results
        predictor.export_results(result, f"rna_results_{seq_name.replace(' ', '_')}.json")
        
        # Compare multiple structures
        if input("\nWould you like to compare multiple sequences? (y/n): ").lower() == 'y':
            comparison_seqs = [examples[k] for k in list(examples.keys())[:3]]
            comparison_labels = list(examples.keys())[:3]
            
            print("\nComparing structures...")
            comparison_df = predictor.compare_structures(comparison_seqs, comparison_labels)
            print("\n" + "=" * 70)
            print("STRUCTURE COMPARISON".center(70))
            print("=" * 70)
            print(comparison_df.to_string(index=False))
            
            # Save comparison
            comparison_df.to_csv("rna_comparison.csv", index=False)
            print("\nComparison saved to rna_comparison.csv")
        
    except Exception as e:
        print(f"\nError: {e}")
        return
    
    print("\n" + "=" * 70)
    print("Analysis complete!".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
