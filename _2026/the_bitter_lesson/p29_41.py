rom manimlib import *
from tqdm import tqdm
import re
from pathlib import Path

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' #6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/alpha_go_self_play')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/human_human_kgs-19-2015')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/games_with_videos')
games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/less_wrong_reverse_engineer')



size = 19  # 19x19 board
padding = 0.5
board_width = 8 # Total width in Manim units
step = board_width / size



def parse_sgf(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Regex to find B[xx] or W[xx]
    moves = re.findall(r"([BW])\[([a-s]{2})\]", content)
    
    parsed_moves = []
    for color_char, coords in moves:
        # Convert letters to 0-18 integers
        x = ord(coords[0]) - ord('a')
        y = ord(coords[1]) - ord('a')
        # Flip Y because SGF is top-down, Manim is bottom-up
        y = 18 - y 
        
        color = BLACK if color_char == 'B' else WHITE
        parsed_moves.append((x, y, color))
    return parsed_moves



def create_stone(x, y, color=BLACK):
    """Create a 3D stone at the given grid position."""
    stone_radius=step*0.45
    pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
    squash = 0.3  # How flat the stone is
    
    if color == BLACK:
        stone = Sphere(radius=stone_radius)
        # stone.set_color("#1a1a1a")
        # stone.set_color("#222222")
        stone.set_color(BLACK)
        # stone.set_shading(0.3, 0.8, 0.2)  # ambient, diffuse, specular
        stone.set_shading(0.1, 0.4, 0.1)
    else:  # white
        stone = Sphere(radius=stone_radius)
        stone.set_color("#B0B0B0")
        # stone.set_shading(0.5, 0.7, 0.3)
        stone.set_shading(0.7, 0.9, 0.9)

    # Squash in z-direction to make it stone-shaped
    stone.scale([1, 1, squash])
    
    # Lift it slightly so it sits on the board
    pos[2] = stone_radius * squash
    stone.move_to(pos)
    
    return stone




class P29_41(InteractiveScene):
    def construct(self): 
        '''
        Ok leaning towards doing a bit more in manim than I first thought, 
        mostly becuase I think there's a path that's not so bad
        Let me see how this goes/feels, and I can fall back to less Manim 
        if I need to. 
        '''

        





        self.wait(20)
        self.embed()

















