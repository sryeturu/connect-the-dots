

def parse_config(config_file_name):
    config = []
    
    with open(config_file_name) as file:    
        cur = {}

        for line in file:      
            line = line.strip()      
            
            if len(line) == 0 or line[0] == '#':
                continue

            elif line[0] == '[':           
                if len(cur) != 0:
                    config.append(cur)
                    cur = {}

                cur['type'] = line[1:-1]       

            else:
                key, value = [x.strip() for x in line.split('=')]
                
                if ',' in value:
                    value = value.split(',')
                
                cur[key] = value
    
    config.append(cur)
    model_config = config.pop(0)
    
    return model_config, config