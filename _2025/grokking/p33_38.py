from manimlib import *
from functools import partial

from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors
from tqdm import tqdm

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

def viridis_hex(value, vmin, vmax):
    """
    Map a scalar `value` in [vmin, vmax] to a Viridis color (hex string).
    """
    # Normalize into [0,1]
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    # Get RGBA from viridis
    rgba = cm.viridis(norm(value))
    # Convert to hex
    return colors.to_hex(rgba)

def black_to_tan_hex(value, vmin, vmax=1):
    """
    Map a scalar `value` in [vmin, vmax] to a color from black to FRESH_TAN (#dfd0b9).
    """
    cmap = colors.LinearSegmentedColormap.from_list('black_tan', ['#000000', '#dfd0b9'])
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return colors.to_hex(cmap(norm(value)))

def softmax_with_temperature(logits, temperature=1.0, axis=-1):
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=axis, keepdims=True))
    return exp_scaled / np.sum(exp_scaled, axis=axis, keepdims=True)




def draw_inputs(self, activations, all_svgs, reset=False, example_index=0, wait=0.0):

    #Borders and fills
    input_mapping_1a=[[0, 1], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]
    input_mapping_1b=[[19, 20], [21, 22]]
    input_mapping_2a=[[23, 24], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41]]
    input_mapping_2b=[[42, 43], [44, 45]]
    input_mapping_3a=[[46, 47], [51, 52], [53, 54], [55, 56], [57, 58], [59, 60], [61, 62], [63, 64]]
    input_mapping_3b=[[65, 66], [67, 68]]


    #Color inputs
    for mapping, activations_index, offset in zip([input_mapping_1a, input_mapping_1b, input_mapping_2a, input_mapping_2b, input_mapping_3a, input_mapping_3b], 
                                          [0, 0, 1, 1, 2, 2], [0, 112, 0, 112, 0, 112]):
        for i, idx in enumerate(mapping):
            if i+offset == activations['x'][example_index][activations_index]:
                all_svgs[2][idx[0]].set_color(FRESH_TAN)
            else:
                all_svgs[2][idx[0]].set_color(BLACK)
            if reset:
                all_svgs[2][idx[0]].set_color(BLACK)
        if wait!=0.0: self.wait(wait)

def draw_embeddings(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):

    embedding_fill_indices_1=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    embedding_fill_indices_2=[28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
    embedding_fill_indices_3=[53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73]
    # vmin=np.min(activations['blocks.0.hook_resid_pre'])*0.9 
    # vmax=np.max(activations['blocks.0.hook_resid_pre'])*0.9
    for i, indices in enumerate([embedding_fill_indices_1, embedding_fill_indices_2, embedding_fill_indices_3]):
        vmin=np.min(activations['blocks.0.hook_resid_pre'][example_index][i])*1.5 #Scaling by column
        vmax=np.max(activations['blocks.0.hook_resid_pre'][example_index][i])*1.5
        for j, idx in enumerate(indices):
            # c=viridis_hex(activations['blocks.0.hook_resid_pre'][example_index, 0, i], vmin, vmax)
            c=colormap(activations['blocks.0.hook_resid_pre'][example_index, i, j], vmin, vmax)
            # print(activations['blocks.0.hook_resid_pre'][example_index, 0, j])
            all_svgs[4][idx].set_color(c)
            if reset: all_svgs[4][idx].set_color(BLACK)
            if wait!=0.0: self.wait(wait)


def draw_attention_values(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):

        ## Attention - Values - mapping again is hacky here, technicall first layer should be same for all
        ## Keep moving for now.
        vmin=np.min(activations['blocks.0.attn.hook_v'][example_index])*0.25 #Scaling
        vmax=np.max(activations['blocks.0.attn.hook_v'][example_index])*0.25
        value_fill_indices_1=[0, 2, 4, 6, 8]
        value_fill_indices_2=[10, 12, 14, 16, 18]
        value_fill_indices_3=[20, 22, 24, 26, 28]
        value_fill_indices_4=[30, 32, 34, 36, 38]

        for head_id, indices in enumerate([value_fill_indices_1, value_fill_indices_2, value_fill_indices_3, value_fill_indices_4]):
            for j, idx in enumerate(indices):
                c=colormap(activations['blocks.0.attn.hook_v'][example_index, head_id, 1, j], vmin, vmax)
                all_svgs[12][idx].set_color(c)
                if reset: all_svgs[4][idx].set_color(BLACK)
                if wait!=0.0: self.wait(wait)

def draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):
        #Attention - attention patterns - need to be more precise with these!
        vmin=0
        vmax=0.8
        attn_fill_indices=[[0,0], [1,0], [1,1], [2,0], [2,1], [2,2]] #Indices to sample matrix at
        head_id=0
        for head_id, offset in enumerate([0, 6, 12, 18]):
            for j, idx in enumerate(attn_fill_indices):
                a=activations['blocks.0.attn.hook_attn'][example_index, head_id, idx[0], idx[1]]
                c=black_to_tan_hex(a, vmin, vmax)
                all_svgs[13][offset+j].set_color(c)
                if reset: all_svgs[13][offset+j].set_color(BLACK)
                self.remove(all_svgs[7]); self.add(all_svgs[7]) #Stacking for grid
                if wait!=0.0: self.wait(wait)

def draw_mlp_1(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):
    #MLP layer 1 (565, 3, 128) 
    vmin=np.min(activations['blocks.0.hook_resid_mid'][example_index])*0.85 #Scaling
    vmax=np.max(activations['blocks.0.hook_resid_mid'][example_index])*0.85
    mlp_indices_1=[0, 2, 4, 6, 8, 10, 12]
    for i, idx in enumerate(mlp_indices_1):
        c=black_to_tan_hex(activations['blocks.0.hook_resid_mid'][example_index, 2, i], vmin, vmax)
        all_svgs[9][idx].set_color(c)
        if reset: all_svgs[9][idx].set_color(BLACK)
        if wait!=0.0: self.wait(wait)

def draw_mlp_2(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):
    #MLP Layer 2 (565, 3, 512)
    vmin=np.min(activations['blocks.0.mlp.hook_pre'][example_index])*0.85 #Scaling
    vmax=np.max(activations['blocks.0.mlp.hook_pre'][example_index])*0.85
    mlp_indices_2=[14, 16, 18, 20, 22, 24, 26, 28, 30]
    for i, idx in enumerate(mlp_indices_2):
        c=black_to_tan_hex(activations['blocks.0.mlp.hook_pre'][example_index, 2, i], vmin, vmax)
        all_svgs[9][idx].set_color(c)
        if reset: all_svgs[9][idx].set_color(BLACK)
        if wait!=0.0: self.wait(wait)

def draw_mlp_3(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):
    # MLP Layer 3 (565, 3, 128) 
    vmin=np.min(activations['blocks.0.hook_mlp_out'][example_index])*0.85 #Scaling
    vmax=np.max(activations['blocks.0.hook_mlp_out'][example_index])*0.85
    mlp_indices_3=[32, 34, 36, 38, 40, 42, 44]
    for i, idx in enumerate(mlp_indices_3):
        c=black_to_tan_hex(activations['blocks.0.hook_mlp_out'][example_index, 2, i], vmin, vmax)
        all_svgs[9][idx].set_color(c)
        if reset: all_svgs[9][idx].set_color(BLACK)
        if wait!=0.0: self.wait(wait)



def draw_logits(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex, temperature=25.0):

        #Logits or probs (565, 113)
        logit_indices_1=[3, 5, 7, 9, 11, 13, 15, 17]
        logit_indices_2=[19, 21]

        #Don't love probs or logits, how about temperature?
        probs_sortof=softmax_with_temperature(activations['logits'][example_index], temperature=temperature, axis=0)


        vmin=np.min(probs_sortof)*1.0 #Scaling
        vmax=np.max(probs_sortof)*1.0

        for i, idx in enumerate(logit_indices_1):
            c=black_to_tan_hex(probs_sortof[i], vmin, vmax)
            all_svgs[11][idx].set_color(c)
            if reset: all_svgs[11][idx].set_color(BLACK)
            if wait!=0.0: self.wait(wait)

        for i, idx in enumerate(logit_indices_2):
            c=black_to_tan_hex(probs_sortof[i+111], vmin, vmax)
            all_svgs[11][idx].set_color(c)
            if reset: all_svgs[11][idx].set_color(BLACK)
            if wait!=0.0: self.wait(wait)


svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/grokking/graphics/to_manim')
data_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')


class P33_38(InteractiveScene):
    def construct(self): 

        #Pick up (mostly) where we left off on 31. 

        p=113

        svg_files=list(sorted(svg_dir.glob('*network_to_manim*')))

        with open(data_dir/'final_model_activations_sample.p', 'rb') as f:
            activations = pickle.load(f)

        all_svgs=Group()
        for svg_file in svg_files[1:17]: #Expand if I add more artboards
            svg_image=SVGMobject(str(svg_file))
            all_svgs.add(svg_image[1:]) #Thowout background

        all_svgs.scale(6.0) #Eh?

        draw_inputs(self, activations, all_svgs, reset=True, example_index=0)
        draw_attention_patterns(self, activations, all_svgs, reset=True, example_index=0)
        
        #MLP weights
        np.random.seed(5)
        R=np.random.uniform(0.3, 0.75, len(all_svgs[8]))
        for i in range(len(all_svgs[8])):
            all_svgs[8][i].set_opacity(R[i])


        #Ok want to pick up with the same activation structure we left off with!
        example_index=0
        self.frame.reorient(0, 0, 0, (0.02, -0.06, 0.0), 7.58)

        draw_inputs(self, activations, all_svgs, reset=False, example_index=example_index, wait=0)
        draw_embeddings(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        draw_attention_values(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_1(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_2(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_3(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_logits(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex, temperature=25.0)
        self.remove(all_svgs[7]); self.add(all_svgs[7])

        self.add(all_svgs[:15])

        # Start p28
        # Ok ok so it would be dope to lower the opacity (or maybe just remove?) 
        # everything after the middle MLP layer. 

        p28_fade_group=Group(all_svgs[14][9],
                            all_svgs[14][:3], 
                            all_svgs[10],
                            all_svgs[11],
                            all_svgs[0][7:14],
                            all_svgs[0][-1],
                            all_svgs[8][-105:],
                            all_svgs[9][-14:],
                            all_svgs[14][-20:])


        axes=VGroup()
        for i in range(7):
            a = Axes(
                x_range=[0, 1.0, 1],
                y_range=[-1.0, 1.0, 1],
                width=2.4,
                height=0.8,
                axis_config={
                    "color": CHILL_BROWN,
                    "include_ticks": False,
                    "include_numbers": False,
                    "include_tip": True,
                    "stroke_width":1.2,
                    "tip_config": {"width":0.02, "length":0.02}
                    }
                )
            axes.add(a)
        axes.arrange(DOWN, buff=0.2)
        axes.move_to([4.8, 0, 0])


        all_pts=VGroup()
        neuron_indices=[0, 3, 4, 2, 11, 5, 9]
        for i, neuron_idx in enumerate(neuron_indices):
            neuron_average=activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx].mean()
            neuron_max=np.max(np.abs(activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx]-neuron_average))*1.0 #Might want to bring down
            dese_pts=VGroup()
            for j in range(p):
                x = j / p
                y = (activations['blocks.0.mlp.hook_pre'][j, 2, neuron_idx]-neuron_average)/neuron_max
                pt = Dot(axes[i].c2p(x, y), radius=0.02, color=FRESH_TAN, stroke_width=0)
                pt.set_color(viridis_hex(j, 0, p))
                dese_pts.add(pt)
            all_pts.add(dese_pts)

        axes_2 = VGroup()
        for i in range(49):  # 7 × 7
            a = Axes(
                x_range=[-1.0, 1.0, 1],
                y_range=[-1.0, 1.0, 1],
                width=0.8,
                height=0.8,
                axis_config={
                    "color": CHILL_BROWN,
                    "include_ticks": False,
                    "include_numbers": False,
                    "include_tip": True,
                    "stroke_width": 1.2,
                    "tip_config": {"width": 0.02, "length": 0.02},
                },
            )
            axes_2.add(a)

        axes_2.arrange_in_grid(n_rows=7, n_cols=7, buff=0.2)
        axes_2.move_to([10, 0, 0])
        
        all_pts_2=VGroup()
        for i, neuron_idx_1 in enumerate(neuron_indices):
            for k, neuron_idx_2 in enumerate(neuron_indices):
                dese_pts=VGroup()
                neuron_average_x=activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_1].mean()
                neuron_average_y=activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_2].mean()
                neuron_max_x=np.max(np.abs(activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_1]-neuron_average_x))*1.0 #Might want to bring down
                neuron_max_y=np.max(np.abs(activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_2]-neuron_average_y))*1.0 #Might want to bring down
                for j in range(p):
                    x = (activations['blocks.0.mlp.hook_pre'][j, 2, neuron_idx_2]-neuron_average_y)/neuron_max_y
                    y = (activations['blocks.0.mlp.hook_pre'][j, 2, neuron_idx_1]-neuron_average_x)/neuron_max_x
                    pt = Dot(axes_2[len(neuron_indices)*i+k].c2p(x, y), radius=0.02, stroke_width=0)
                    pt.set_color(viridis_hex(j, 0, p))
                    dese_pts.add(pt)
                all_pts_2.add(dese_pts)

        self.frame.reorient(0, 0, 0, (7.62, 0.06, 0.0), 7.58)
        self.remove(p28_fade_group)



        self.add(axes, axes_2, all_pts, all_pts_2)
        self.add(all_svgs[15])
        self.wait()

        self.play(self.frame.animate.reorient(0, 0, 0, (6.47, 1.97, 0.0), 3.63), 
                  FadeOut(all_pts_2), FadeOut(axes_2), run_time=5)
        self.wait()



        svg_files=list(sorted(svg_dir.glob('*fft_plots*')))
        fft_svgs=Group()
        for svg_file in svg_files[1:]: #Expand if I add more artboards
            svg_image=SVGMobject(str(svg_file))
            fft_svgs.add(svg_image[1:]) #Throwout background

        fft_svgs.scale(1.8) 
        fft_svgs.move_to([7.7, 1.96, 0])

        self.wait()
        self.play(Write(fft_svgs[0]))
        self.play(Write(fft_svgs[1]))
        self.wait()

        self.play(Write(fft_svgs[2]))
        self.wait()

        #Now load up numpy arrays so we can plot sine waves!
        fits=[]
        for i in range(3):
            fits.append(np.load(data_dir/'ffts_1'/('fit_'+str(i)+'.npy')))


        # Draw smooth YELLOW curves on the first 3 axes
        curves = VGroup()
        N = len(fits[0])
        xs = np.linspace(0, 1.0, N)  # x in [0, 1]

        for i in range(3):  # first 3 axes
            y_vals = fits[i]
            if i==1: y_vals=y_vals/3.0 #No idea why I need this, going to keep moving for now. 

            ##normalize if needed to keep in [-1, 1]
            # y_mean = y_vals.mean()
            # y_max = np.max(np.abs(y_vals - y_mean))
            # y_vals = (y_vals - y_mean) / (y_max + 1e-8)

            pts = [axes[i].c2p(xs[j], y_vals[j]) for j in range(N)]

            curve = VMobject(stroke_width=3)
            curve.set_points_smoothly(pts)
            curve.set_color(YELLOW)
            curves.add(curve)

        wave_label_1 = Tex(r'A_1 \cos \big( \frac{8 \pi }{113} x + \phi_1 \big)')
        wave_label_1.set_color(YELLOW)
        wave_label_1.scale(0.28)
        wave_label_1.move_to([4.8, 3.45, 0])

        wave_label_2 = Tex(r'A_2 \cos \big( \frac{8 \pi }{113} x + \phi_2 \big)')
        wave_label_2.set_color(YELLOW)
        wave_label_2.scale(0.28)
        wave_label_2.move_to([4.8, 2.4, 0])

        wave_label_3 = Tex(r'A_3 \cos \big( \frac{6 \pi }{113} x + \phi_3 \big)')
        wave_label_3.set_color(YELLOW)
        wave_label_3.scale(0.28)
        wave_label_3.move_to([4.8, 1.45, 0])

        self.wait()
        self.play(ShowCreation(curves), lag_ration=0.8, run_time=4)
        self.add(wave_label_1, wave_label_2, wave_label_3) #Simple add here I think!
        self.wait()

        # self.remove(wave_label_1, wave_label_2, wave_label_3)

        # self.add(curves)
        # self.remove(curves)

        # self.add(fft_svgs)
        # self.remove(fft_svgs)


        # self.frame.reorient(0, 0, 0, (6.47, 1.97, 0.0), 3.63)

        
        self.add(p28_fade_group); 
        self.remove(all_svgs[9]); self.add(all_svgs[9])
        self.remove(all_svgs[7]); self.add(all_svgs[7]) #Occlusions bro


        self.wait(20)
        self.embed()

            
            

























