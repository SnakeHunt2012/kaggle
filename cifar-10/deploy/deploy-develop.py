from os import system
from itertools import product

def replace(source_file, target_file, source_string, target_string):
    pass

def main():
    # names
    rate_name = "###Learning Rate###"
    batch_name = "###Batch Size###"
    hidden_name = "###Hidden Units###"
    L2_name = "###L2 Squared Weight###"
    activation_name = "###Activation###"
    
    # parameter_list
    rate_list = ["0.00000010", "0.00000001"]
    batch_list = ["500", "1000", "1500"]
    hidden_list = ["3000", "7000", "9000", "11000"]
    L2_list = ["1", "5", "10", "50", "100"]
    activation_list = ["tanh"]
    parameter_list = product(rate_list,
                             batch_list,
                             hidden_list,
                             L2_list,
                             activation_list)
    # iplist
    ip_list = []
    with open("ip.list", 'r') as ip_file:
        raw_ip_list = ip_file.readlines()
    for raw_ip in raw_ip_list:
        ip = raw_ip.strip()
        ip_list.append(ip)
    # remove current machine from the list
    ip_list.remove("10.101.165.148")
    # deploy
    ip_index = 0
    ip_len = len(ip_list)
    deploy_log_list = []
    for rate, batch, hidden, L2, activation in parameter_list:
        # 1. prepare instance name
        instance_name = rate + "_" + \
                        batch + "_" + \
                        hidden + "_" + \
                        L2 + "_" + \
                        activation
        # 2. prepare config file
        source_path = "template.cfg"
        config_dir = "config"
        target_path = config_dir + "/" + \
                      instance_name + ".cfg"
        target_text = []
        with open(source_path, 'r') as source_file:
            source_text = source_file.readlines()
        for line in source_text:
            # replace learning rate
            line = line.replace(rate_name, rate)
            # replace batch size
            line = line.replace(batch_name, batch)
            # replace hidden units
            line = line.replace(hidden_name, hidden)
            # replace L2 squared weight
            line = line.replace(L2_name, L2)
            # replace activation
            line = line.replace(activation_name,
                                "\"" + activation + "\"")
            # append line
            target_text.append(line)
        with open(target_path, 'w') as target_file:
            target_file.writelines(target_text)
        # 3. prepare target_ip
        target_ip = ip_list[ip_index % ip_len]
        ip_index += 1
        # 4. log
        deploy_log_list.append((instance_name, target_ip))
        # 5. launch
        system_command = "./add-develop.sh" + ' ' + \
                         target_path + ' ' + \
                         target_ip + ' ' + \
                         instance_name
        system(system_command)
    # record deploy log
    with open("deploy-log.csv", "w") as log_file:
        log_file.write("instance_name,target_ip\n")
        for instance_name, target_ip in deploy_log_list:
            log_file.write(instance_name + ',' + target_ip + '\n')
        
    

if __name__ == "__main__":
    main()
