import json
import sys

def fix_notebook(notebook_path, output_path):
    """
    Fix a Jupyter notebook by adding empty 'outputs' field to code cells that don't have it.
    
    Parameters:
    -----------
    notebook_path : str
        Path to the input notebook
    output_path : str
        Path to save the fixed notebook
    """
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Fix code cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'outputs' not in cell:
            cell['outputs'] = []
    
    # Save the fixed notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Fixed notebook saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_notebook.py input_notebook.ipynb output_notebook.ipynb")
        sys.exit(1)
    
    fix_notebook(sys.argv[1], sys.argv[2])