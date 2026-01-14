import modal

app = modal.App("check-checkpoint")

checkpoint_vol = modal.Volume.from_name("viral-checkpoints", create_if_missing=False)

@app.function(
    volumes={"/checkpoints": checkpoint_vol},
)
def inspect_volume():
    """Check what's actually in the checkpoint volume"""
    import os
    
    print("="*60)
    print("CHECKPOINT VOLUME CONTENTS")
    print("="*60)
    
    # Check root
    if os.path.exists("/checkpoints"):
        print("\n📁 /checkpoints:")
        for item in os.listdir("/checkpoints"):
            path = os.path.join("/checkpoints", item)
            if os.path.isdir(path):
                print(f"  📁 {item}/")
                # List contents of subdirectories
                try:
                    sub_items = os.listdir(path)
                    for sub in sub_items[:10]:  # Show first 10
                        print(f"     - {sub}")
                    if len(sub_items) > 10:
                        print(f"     ... and {len(sub_items) - 10} more files")
                except:
                    pass
            else:
                size = os.path.getsize(path)
                print(f"  📄 {item} ({size:,} bytes)")
    else:
        print("❌ /checkpoints doesn't exist!")
    
    # Check final directory specifically
    final_path = "/checkpoints/final"
    print(f"\n📦 Checking {final_path}:")
    if os.path.exists(final_path):
        files = os.listdir(final_path)
        print(f"✅ Found {len(files)} files:")
        for f in files:
            fpath = os.path.join(final_path, f)
            size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
            print(f"  - {f} ({size:,} bytes)")
        
        # Check for required files
        required = ['adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json']
        print("\n🔍 Required files check:")
        for req in required:
            status = "✅" if req in files else "❌"
            print(f"  {status} {req}")
    else:
        print(f"❌ {final_path} doesn't exist!")
    
    return True

@app.local_entrypoint()
def main():
    result = inspect_volume.remote()
    print("\n✅ Inspection complete")