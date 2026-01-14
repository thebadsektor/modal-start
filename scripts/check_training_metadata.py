import modal
import json

app = modal.App("check-metadata")
checkpoint_vol = modal.Volume.from_name("viral-checkpoints", create_if_missing=False)

@app.function(volumes={"/checkpoints": checkpoint_vol})
def read_metadata():
    """Read training metadata to see what happened"""
    
    metadata_path = "/checkpoints/final/training_metadata.json"
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("="*60)
        print("TRAINING METADATA")
        print("="*60)
        print(json.dumps(metadata, indent=2))
        print("="*60)
        
        # Check if training was successful
        final_loss = metadata.get('final_loss', 999)
        if final_loss < 0.5:
            print("✅ Training completed successfully (low loss)")
        elif final_loss < 1.5:
            print("⚠️ Training completed but loss is moderate")
        else:
            print("❌ Training loss is high - model may not have learned well")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return None

@app.local_entrypoint()
def main():
    metadata = read_metadata.remote()