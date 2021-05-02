import random
import argparse
from pathlib import Path



def _generate_random_taxonomy(sizes:list, seed=42):
    """
    args - sizes: the size you want each height to be. 
    """
    
    # opening up the file with all instruments
    f = open('music_trees/assets/taxonomies/all-instruments.yaml')
    # splitting the list so that instruments aare their own elements
    instruments = f.read().split('\n')
    
    # finding the number of instruments per group
    group_sizes = []
    inst_per_group = len(instruments)

    for value in sizes:
        inst_per_group //= value
        group_sizes.append(inst_per_group)
    
    # setting the random seed
    random.seed(seed)
    
    # random shuffle of the list
    random.shuffle(instruments)
    
    # track the number of groups we have built, and store all grouops in a dict named taxonomy
    # total_groups must be a closure which can modify its internal state
    total_groups = [0]
    
    #DEBUG
    print(f'sizes: {sizes}\ngroup_sizes: {group_sizes}')
    
    def random_taxos_helper(sizes:list, group_sizes:list, instruments:list, taxo={}):
        # base case we are at final sub group we want this group to contain all remaining instruments
        if not sizes:
            return instruments
        
        # if we have more groups to build iterate through the number of groups and continue recursively 
        else:
            # get the size of each subgroup we will build
            curr_group_size = group_sizes[0]
            
            for i in range(sizes[0]):
                # special case, if we are at the last group then grab the remaining instruments
                # this will prevent any insturments from being left out
                if i == sizes[0] - 1:
                    subset = instruments[(i * curr_group_size):]
                
                # grab the segment of instruments for this group
                else:
                    subset = instruments[(i * curr_group_size):((i + 1) * curr_group_size)]
                
                # update our dict
                group_key = f'Group {total_groups[0]}'
                total_groups[0] += 1
                result = random_taxos_helper(sizes[1:], group_sizes[1:], subset, taxo={})
                taxo[group_key] = result
        
        return taxo
    
    # return the result of the helper
    return random_taxos_helper(sizes, group_sizes, instruments)

def _write_taxonomy(file_num:int, path:str, taxo:dict):
    """ Helper function which writes the taxonomy to a file in the specifed path. """
    f = open(path + '/' +  f'random-taxonomy-{file_num}.yaml', "w")
    taxonomy = ['']
    def _write_helper(taxo, i=0):
        # based case we've hit a list of instruments
        if  isinstance(taxo, list):
            for inst in taxo:
                cur_chunck = '\t'*i + f'{inst}\n'
                taxonomy[0] += cur_chunck
        else:
            for key in taxo.keys():
                group_name = '\t'*i + f'{key}:\n'
                taxonomy[0] += group_name
                _write_helper(taxo[key], i+1)
    
    _write_helper(taxo)
    f.write(taxonomy[0])

def generate_random_taxonomies(num_taxonomies:int, sizes:list, size_type='single', seed=42):
    """
    a wrapper function for generating random taxonomies. Generates num_taxonomies random taxonomies.
    
    args:
        num_taxonomies (list) - the number of random taxonomies to generate
        sizes (list) - list of sizes, can be a list of list or a single list
        size_type (string) - two option 'single' or 'multi', specification for sizes
                                'single' - use one list of sizes for all taxnomies (sizes is 1D list)
                                'multi' - use multiple lists for generating taxonomies 
                                    (sizes can be at most num_taxonomies x # of instruments)
        seed (int) - base send to use when generating the taxonomies
    side-effects:
        The function will create a directory in music-trees/music-trees/assets/taxonomies/ called random taxonomies.
        This directory will contain the randomly generated taxonomies
     """
    # creating the directory for the random taxonomies
    results_dir = 'music_trees/assets/taxonomies/random-taxonomies'
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # iterate through the number of taxonomies we want to create, call _generate_random_taxonomy for each new taxonomy
    # use seed + i to create a new random seed
    for i in range(num_taxonomies):
        # start by getting the size for taxonomy i
        if size_type == 'single':
            taxo_size = sizes
        elif size_type == 'multi':
            # use % here incase len(sizes) < num_taxonomies
            taxo_size = sizes[i%len(sizes)]
        else:
            # basice value error
            raise ValueError(f'"{size_type}" is an invalid string for size_type. Please use "single" or "multi".')
       
        print(f'Generating random-taxonomy-{i}')

        # call helper function to get a dict for the current random taxonomy
        curr_taxo = _generate_random_taxonomy(taxo_size, seed+i)
        
        # write the current taxonomy as a yaml file using helper function
        _write_taxonomy(i, results_dir, curr_taxo)

        # adding some space for print statements
        print('-'*100,'\n')
    
def valid_parentheses(arg_list:str):
    stack = []
    for char in arg_list:
        if char == '[':
            stack.append('[')
        if char == ']':
            stack.pop()

    if len(stack) != 0:
        raise ValueError(f'"{arg_list}" is an invalid list for --sizes')

# TODO clean this up
def _convert_to_list(lst_str):

    valid_parentheses(lst_str)
    i = 0
    lst = []
    while i < len(lst_str):
        
        if lst_str[i] == '[' and i > 0:
            sub_list = []
            while i < len(lst_str) and lst_str[i] != ']':
                if lst_str[i].isnumeric():
                    sub_list.append(int(lst_str[i]))
                i += 1
            lst.append(sub_list)
            
        elif lst_str[i].isnumeric():
            lst.append(int(lst_str[i]))
            i += 1
        else:
            i += 1
    return lst





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add training script arguments
    parser.add_argument('--num_taxonomies', type=int, required=True,
                        help='number of taxonomies to generate.')

    parser.add_argument('--sizes', type=str, required=True,
                        help='sizes for the taxonomies to be generated.')
    # TODO get rid of this, it's implicit from the sizes 
    # i.e if sizes[0] is a list then multi else single
    parser.add_argument('--size_type', type=str, required=True,
                        help='single or multi specification for the sizes argument.')

    args = parser.parse_args()

    num_taxonomies = int(args.num_taxonomies)
    sizes = _convert_to_list(args.sizes)
    size_type = args.size_type

    generate_random_taxonomies(num_taxonomies, sizes, size_type)