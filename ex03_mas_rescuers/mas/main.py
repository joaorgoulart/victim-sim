import os
import sys
import time

from explorer import Explorer
from rescuer import Rescuer

## importa classes
from vs.environment import Env


def main(data_folder_name, config_ag_folder_name):
    try:
        # Set the path to config files and data files for the environment
        current_folder = os.path.abspath(os.getcwd())
        config_ag_folder = os.path.join(current_folder, config_ag_folder_name)
        data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))
        
        # Instantiate the environment
        env = Env(data_folder)
        
        # Instantiate master_rescuer
        # This agent unifies the maps and instantiate other 3 agents
        rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
        master_rescuer = Rescuer(env, rescuer_file, 4)   # 4 is the number of explorer agents

        # Explorer needs to know rescuer to send the map 
        # that's why rescuer is instatiated before
        for exp in range(1, 5):
            filename = f"explorer_{exp:1d}_config.txt"
            explorer_file = os.path.join(config_ag_folder, filename)
            Explorer(env, explorer_file, master_rescuer)

        # Run the environment simulator
        env.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_300v_90x90")
        # Correct the path to reference cfg_1 in ex03_mas_rescuers
        config_ag_folder_name = os.path.join('ex03_mas_rescuers', "cfg_1")
                
    main(data_folder_name, config_ag_folder_name)
