def load(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    metadata = lines[0].strip()
    data = {}
    label = None
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("<"):
            label = line[1:]
            if label.endswith("_metadata"):
                data[label] = {}
            else:
                data[label] = []
        elif line.endswith(">"):
            label = None
        elif label is not None:
            if label.endswith("_metadata"):
                key, value = line.split("=")
                value = float(value.strip())
                data[label][key.strip()] = value
            else:
                values = list(map(float, line.strip().split()))
                data[label].append(values)

    return metadata, data
