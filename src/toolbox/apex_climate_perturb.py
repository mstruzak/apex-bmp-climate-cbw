import os
import glob


# Defined T and P Perturbations for APEX climate files: .dly, .hly, .wp1
# used for sobol analysis 

# .dly: cols 21-32 = Tmax, Tmin; cols 33-38 = Precip
def perturb_dly(filepath, temp_delta, precip_factor, output_dir=None):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if len(line) < 38:
            new_lines.append(line)
            continue
        
        # extract and perturb values
        tmax = float(line[20:26]) + temp_delta
        tmin = float(line[26:32]) + temp_delta
        prcp = float(line[32:38]) * precip_factor
        
        # reconstruct line preserving format
        new_line = line[:20] + f"{tmax:6.2f}{tmin:6.2f}{prcp:6.2f}" + line[38:]
        new_lines.append(new_line)
    
    outpath = _get_output_path(filepath, temp_delta, precip_factor, output_dir)
    with open(outpath, 'w') as f:
        f.writelines(new_lines)
    return outpath

def perturb_hly(filepath, precip_factor, temp_delta=None, output_dir=None):
    """Perturb precipitation (last 10 cols) in .hly file. temp_delta unused but kept for consistent API."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        stripped = line.rstrip('\r\n')
        if len(stripped) < 20:
            new_lines.append(line)
            continue
        
        # Precip is last 10 characters before line ending
        prcp = float(stripped[-10:]) * precip_factor
        new_line = stripped[:-10] + f"{prcp:10.3f}" + line[len(stripped):]
        new_lines.append(new_line)
    
    outpath = _get_output_path(filepath, temp_delta, precip_factor, output_dir)
    with open(outpath, 'w') as f:
        f.writelines(new_lines)
    return outpath

def perturb_wp1(filepath, temp_delta, precip_factor, output_dir=None):
    """Perturb temperature (lines 3-4) and precipitation (line 7) in .wp1 file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i, line in enumerate(lines):
        line_num = i + 1
        
        if line_num in [3, 4]:  # TMX, TMN lines
            new_line = _perturb_wp1_line(line, temp_delta, add=True)
        elif line_num == 7:  # PRCP line
            new_line = _perturb_wp1_line(line, precip_factor, add=False)
        else:
            new_line = line
        new_lines.append(new_line)
    
    outpath = _get_output_path(filepath, temp_delta, precip_factor, output_dir)
    with open(outpath, 'w') as f:
        f.writelines(new_lines)
    return outpath

def _perturb_wp1_line(line, value, add=True):
    """Perturb 12 monthly values in a .wp1 line, preserving original spacing."""
    stripped = line.rstrip('\r\n')
    parts = stripped.split()
    label = parts[-1]  # e.g., "TMX", "PRCP"
    
    # Perturb the 12 numeric values
    new_vals = []
    for v in parts[:12]:
        val = float(v)
        val = val + value if add else val * value
        new_vals.append(f"{val:10.2f}")
    
    line_ending = line[len(stripped):]
    return ''.join(new_vals) + ' ' + label + line_ending

def _get_output_path(filepath, temp_delta, precip_factor, output_dir):
    """Generate output path with perturbation suffix."""
    base, ext = os.path.splitext(os.path.basename(filepath))
    sign = '+' if temp_delta >= 0 else ''
    suffix = f"_{sign}{temp_delta}T{precip_factor}P"
    filename = base + suffix + ext
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    return os.path.join(os.path.dirname(filepath), filename)

def perturb_directory(input_dir, temp_delta, precip_factor, output_dir=None):
    """Perturb all .dly, .hly, and .wp1 files in a directory."""
    results = []
    
    for ext, func in [('.dly', perturb_dly), ('.hly', perturb_hly), ('.wp1', perturb_wp1)]:
        for filepath in glob.glob(os.path.join(input_dir, f'*{ext}')):
            if ext == '.hly':
                outpath = func(filepath, precip_factor, temp_delta, output_dir)
            else:
                outpath = func(filepath, temp_delta, precip_factor, output_dir)
            results.append(outpath)
            print(f"Created: {outpath}")
    
    return results

if __name__ == "__main__":
    input_dir = # path to existing weather files
    temp_delta = 2
    precip_factor = 1.5
    output_dir = # path to output
    
    perturb_directory(input_dir, temp_delta, precip_factor, output_dir)

