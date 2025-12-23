import yaml

try:
    with open("config.yaml", "r") as f:
        data = yaml.safe_load(f)
        print("✅ YAML is OK!")
        print(data)
except Exception as e:
    print(f"❌ YAML Error: {e}")