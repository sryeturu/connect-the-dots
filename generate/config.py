
def parse_cfg(cfg_path):
 
    all_lines = []
    
    with open(cfg_path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == '' or line[0] == '#':
                continue
            else:
                all_lines.append(line.strip())

            
    def get_key_value(line):
        key, value = line.split('=')

        key = key.strip()
        value = value.strip()
        
        if ',' in value:
            value = value.split(',')
        
        return key, value
    
    cfg = {}
    
    cur_iter = 0
    while  cur_iter < len(all_lines):
        line = all_lines[cur_iter]

        if line[0] == '[':
            collection_name = line[1:-1]

            sub_cfg = {}
            
            cur_iter += 1
            
            while all_lines[cur_iter] != '[end]':
                key, value = get_key_value(all_lines[cur_iter])
                sub_cfg[key] = value
                cur_iter += 1
                
            cfg[collection_name] = sub_cfg
        else:
            key, value = get_key_value(line)
            cfg[key] = value
            
        cur_iter += 1
    return cfg





