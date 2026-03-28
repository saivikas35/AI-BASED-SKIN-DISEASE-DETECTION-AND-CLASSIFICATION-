import numpy as np
import matplotlib.cm as cm

# Simulate what happens in app.py
def test_indexing():
    try:
        heatmap = np.random.rand(192, 192).astype(np.float32)
        heatmap = np.uint8(255 * heatmap)
        
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        
        print(f"jet_colors type: {type(jet_colors)}")
        print(f"heatmap type: {type(heatmap)}")
        
        jet_heatmap = jet_colors[heatmap]
        print(f"jet_heatmap shape: {jet_heatmap.shape}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

test_indexing()
