import modal
import json

app = modal.App("check-data-volume")
data_vol = modal.Volume.from_name("viral-data", create_if_missing=False)

@app.function(volumes={"/data": data_vol})
def inspect_data_volume():
    """Check what training data is in the volume"""
    import os
    
    print("="*60)
    print("DATA VOLUME CONTENTS")
    print("="*60)
    
    if os.path.exists("/data"):
        files = os.listdir("/data")
        print(f"\n📁 /data ({len(files)} files):")
        for f in files:
            path = os.path.join("/data", f)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                print(f"  📄 {f} ({size:,} bytes)")
        
        # Check the training file specifically
        training_files = [f for f in files if 'training' in f and f.endswith('.jsonl')]
        
        if training_files:
            print(f"\n🔍 Found {len(training_files)} training file(s):")
            for tf in training_files:
                filepath = os.path.join("/data", tf)
                print(f"\n📝 {tf}:")
                
                # Count lines
                with open(filepath, 'r') as f:
                    lines = sum(1 for _ in f)
                print(f"   Lines: {lines:,}")
                
                # Show first 2 examples
                print("   First 2 examples:")
                with open(filepath, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 2:
                            break
                        try:
                            data = json.loads(line)
                            print(f"\n   Example {i+1}:")
                            print(f"   Input: {data['input'][:100]}...")
                            print(f"   Output: {data['output'][:100]}...")
                        except:
                            print(f"   Example {i+1}: Failed to parse")
        else:
            print("\n❌ No training files found!")
    else:
        print("❌ /data doesn't exist!")

@app.local_entrypoint()
def main():
    inspect_data_volume.remote()