from manimlib import *
from functools import partial

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' #6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'

data_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/grokking/from_linux/nov_25_1/'
resolution=113

alphas_1=np.linspace(0, 1, resolution) #Crank up here for better spatial resolution I think
def param_surface(u, v, surf_array):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        z = 0.1*surf_array[v_idx, u_idx] #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])


class GrokkingHackingOne(InteractiveScene):
    def construct(self):  

        mlp_hook_pre=np.load(data_dir+'mlp_hook_pre.npy') #Lots of data but seems fast enough?


        neuron_idx=260

        surf_func=partial(param_surface, surf_array=mlp_hook_pre[:,:,2,neuron_idx])
        surface = ParametricSurface(
            surf_func,  
            u_range=[0, 1.0],
            v_range=[0, 1.0],
            resolution=(resolution, resolution),
        )

        self.wait()

        ts = TexturedSurface(surface, data_dir+'activations_'+str(neuron_idx).zfill(3)+'.png')
        ts.set_shading(0.0, 0.1, 0)


        self.add(ts)


        



        # num_lines = 64  # Number of gridlines in each direction
        # num_points = 512  # Number of points per line
        # u_gridlines = VGroup()
        # v_gridlines = VGroup()
        # u_values = np.linspace(-2.5, 2.5, num_lines)
        # v_points = np.linspace(-2.5, 2.5, num_points)
        # for u in u_values:
        #     points = [surf_func(u, v) for v in v_points]
        #     line = VMobject()
        #     line.set_points_smoothly(points)
        #     line.set_stroke(width=1, color=WHITE, opacity=0.15)
        #     u_gridlines.add(line)

        # u_points = np.linspace(-2.5, 2.5, num_points)
        # for v in u_values:  # Using same number of lines for both directions
        #     points = [surf_func(u, v) for u in u_points]
        #     line = VMobject()
        #     line.set_points_smoothly(points)
        #     line.set_stroke(width=1, color=WHITE, opacity=0.15)
        #     v_gridlines.add(line)
        # grids.add(VGroup(u_gridlines, v_gridlines))
        

        self.wait(20)
        self.embed()  