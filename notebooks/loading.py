from pathlib import Path
from collections import Counter
import pandas as pd
from lasso.dyna import D3plot

data_dir = Path("../data")
d3plot_file = data_dir / "d3plot"
dyn_file = data_dir / "child_model.dyn"
key_file = data_dir / "child_entities.key"
k_file = data_dir / "source_nodes.k"

dyn_lines = dyn_file.read_text(errors="ignore").splitlines()
key_lines = key_file.read_text(errors="ignore").splitlines()
k_lines = k_file.read_text(errors="ignore").splitlines()

keyword_files = {
    "child_model": (dyn_file, dyn_lines),
    "entities": (key_file, key_lines),
    "source_node": (k_file, k_lines)
}

keyword_summary = pd.DataFrame([
    {
        "file": name,
        "filename": path.name,
        "lines": len(lines),
        "keywords": sum(line.lstrip().startswith("*") for line in lines),
    }
    for name, (path, lines) in keyword_files.items()
])

print(keyword_summary)

model_features = []
for name, (_, lines) in keyword_files.items():
    keywords = [
        line.strip().upper()
        for line in lines
        if line.lstrip().startswith("*")
    ]
    for keyword, count in Counter(keywords).most_common():
        model_features.append({
            "file": name,
            "keyword": keyword,
            "count": count
        })

model_features = pd.DataFrame(model_features)
model_features.head(30)

d3plot = D3plot(d3plot_file, state_filter = {0, -1})
d3plot_summary = pd.DataFrame({
    "feature": [
        "file",
        "loaded_timesteps",
        "nodes",
        "solid_elements",
        "shell_elements",
        "beam_elements",
        "available_arrays"
    ],
    "value": [
        d3plot_file.name,
        d3plot.n_timesteps,
        d3plot.header.n_nodes,
        d3plot.header.n_solids,
        d3plot.header.n_shells,
        d3plot.header.n_beams,
        len(d3plot.arrays),
    ],
})

print(d3plot_summary)