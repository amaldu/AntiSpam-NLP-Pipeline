from colorama import Fore


def print_and_highlight_diff(orig_text, new_texts):
    """ A simple diff viewer for augmented texts. """
    for orig_text, new_text in zip(orig_text, new_texts):
        orig_split = orig_text.split()
        print("-"*50) 
        print(f"Original: {len(orig_split)}\n{orig_text}")
        print(f"Augmented: {len(new_text.split())}")
        
        for i, word in enumerate(new_text.split()):
            if i < len(orig_split) and word == orig_split[i]:
                print(word, end=" ")
            else:
                print(Fore.RED + word + Fore.RESET, end=" ")
                
        print()