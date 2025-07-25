from manimlib import *
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
from geometric_dl_utils import *


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'


graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are

class refactor_sketch_1(InteractiveScene):
    def construct(self):
        
        #2x2
        # model_path='_2025/backprop_3/models/2_2_1.pth'
        # model = BaarleNet([2,2])
        # model.load_state_dict(torch.load(model_path))
        # viz_scales=[0.25, 0.25, 0.3, 0.3, 0.15]
        # num_neurons=[2, 2, 2, 2, 2]

        #3x3
        model_path='_2025/backprop_3/models/3_3_1.pth'
        model = BaarleNet([3,3])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.1, 0.1, 0.05, 0.05, 0.15]
        num_neurons=[3, 3, 3, 3, 2]

        #8x8
        # model_path='_2025/backprop_3/models/8_8_1.pth'
        # model = BaarleNet([8,8])
        # model.load_state_dict(torch.load(model_path))
        # # viz_scales=[0.1, 0.1, 0.05, 0.05, 0.15]
        # num_neurons=[8,8, 8, 8, 2]

        vertical_spacing=1.5
        horizontal_spacing=3
        colors = [BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]

        # Hmm ok I think I need to compute a different viz scale for each surface probably 
        # I don't want to totally normalize the heights though, ya know?
        # Maybe there's a discrete set of possible viz scales: 
        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=1.0, extent=1)


        surfaces=[]
        surface_funcs=[]
        surface_funcs_no_viz_scale=[]
        for layer_idx in range(len(model.model)):
            s=Group()
            surface_funcs.append([])
            surface_funcs_no_viz_scale.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx])
                surface_func_no_scaling=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=1.0) #adaptive_viz_scales[layer_idx][neuron_idx])
                bent_surface = ParametricSurface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=TexturedSurface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
                ts.set_shading(0,0,0).set_opacity(0.75)
                s.add(ts)
                surface_funcs[-1].append(surface_func)
                surface_funcs_no_viz_scale[-1].append(surface_func_no_scaling)
            surfaces.append(s)

        for layer_idx, sl in enumerate(surfaces):
            for neuron_idx, s in enumerate(sl):
                s.shift([horizontal_spacing*layer_idx-6, 0, vertical_spacing*neuron_idx])
                self.add(s)


        self.frame.reorient(0, 54, 0, (1.41, 1.82, 4.15), 15.71)


        #Optional but kinda nice RuLu intersection planes
        layer_idx=0
        relu_intersections_planes_1=get_relu_intersection_planes(num_neurons[layer_idx], layer_idx, neuron_idx, horizontal_spacing, vertical_spacing)
        self.add(relu_intersections_planes_1)


        #Layer 1 polygons - use surface function to find heights
        layer_idx=1
        layer_1_polygons=get_polygon_corners_layer_1(model)

        #I think i need to use unscaled surface function here so Relus hit at the right spot
        layer_1_polygons_3d=get_3d_polygons_layer_1(layer_1_polygons, surface_funcs_no_viz_scale, num_neurons=num_neurons[layer_idx], layer_idx=1)
        
        #Ok now I need to rescale. 
        scaled_layer_1_polygons_3d=apply_viz_scale_to_3d_polygons(layer_1_polygons_3d, adaptive_viz_scales[layer_idx])

        polygons_vgroup=viz_3d_polygons(scaled_layer_1_polygons_3d, layer_idx, colors=None)
        self.add(polygons_vgroup)
        self.wait()


        #Shadow of layer 1 polygons, basically the regions available to our second layer
        layer_2_polygons=carve_plane_with_relu_joints([o['relu_line'] for o in layer_1_polygons])


        #2d shadow of these regions
        output_poygons_2d=viz_carved_regions_flat(layer_2_polygons, horizontal_spacing, layer_idx, colors=None)
        self.add(output_poygons_2d)


        #Layer 2 linear
        layer_idx=2
        layer_2_polygons_3d=get_3d_polygons(layer_2_polygons, num_neurons[layer_idx], surface_funcs_no_viz_scale, layer_idx)
        scaled_layer_2_polygons_3d=apply_viz_scale_to_3d_polygons(layer_2_polygons_3d, adaptive_viz_scales[layer_idx])


        polygons_vgroup_2=viz_3d_polygons(scaled_layer_2_polygons_3d, layer_idx, colors=None)
        self.add(polygons_vgroup_2)

   
        relu_intersections_planes_2=get_relu_intersection_planes(num_neurons[layer_idx], layer_idx, neuron_idx, horizontal_spacing, vertical_spacing)
        self.add(relu_intersections_planes_2)


        #Layer 2 Relu
        layer_idx=3
        all_polygons, merged_zero_polygons, unmerged_polygons = split_polygons_with_relu(layer_2_polygons_3d)


        #Ok a little clunky to do a post merge like this, but I think this gives some good flexiblity!
        all_polygons_after_merging=copy.deepcopy(merged_zero_polygons)
        for i, o in enumerate(unmerged_polygons):
            all_polygons_after_merging[i].extend(o)


        all_polygons_after_merging_scaled=apply_viz_scale_to_3d_polygons(all_polygons_after_merging, adaptive_viz_scales[layer_idx])

        layer_2_polygons_split_vgroup=viz_3d_polygons(all_polygons_after_merging_scaled, layer_idx, colors=None, color_first_polygon_gray=True)
        self.add(layer_2_polygons_split_vgroup)


        #2D Projection of Layer 2 After Relu Cuts
        #Drop last coords - using unnscaled polgons - dont think it matters?
        all_polygons_after_merging_2d=[]
        for p in all_polygons_after_merging:
            pd2=[o[:,:2] for o in p]
            all_polygons_after_merging_2d.append(pd2)

        layer3_regions_2d = find_polygon_intersections(all_polygons_after_merging_2d)
        output_poygons_2d_2=viz_carved_regions_flat(layer3_regions_2d, horizontal_spacing, layer_idx, colors=None)
        self.add(output_poygons_2d_2)


        # Layer 3 Linear
        # Ok output layer and decision boundary
        # I kinda want a version tha thas all the fun colors and then a yellow and blue version
        layer_idx=4
        # adaptive_viz_scales[layer_idx]=[0.02, 0.02] #Try some manual scale here
        layer_3_polygons_3d=get_3d_polygons(layer3_regions_2d, num_neurons[layer_idx], surface_funcs_no_viz_scale, layer_idx)
        scaled_layer_3_polygons_3d=apply_viz_scale_to_3d_polygons(layer_3_polygons_3d, adaptive_viz_scales[layer_idx])

        polygons_vgroup_3=viz_3d_polygons(scaled_layer_3_polygons_3d, layer_idx, colors=None)
        self.add(polygons_vgroup_3)


        # Ok this is looking dope! 
        # Now I think it's more more layer (that isn't really a layer)
        # With output planes brought together, solid colors, and decision boundary!
        scaled_final_polygons=copy.deepcopy(scaled_layer_3_polygons_3d)
        polygons_vgroup_4a=viz_3d_polygons([scaled_final_polygons[0]], layer_idx=5, colors=[BLUE])
        polygons_vgroup_4b=viz_3d_polygons([scaled_final_polygons[1]], layer_idx=5, colors=[YELLOW])
        self.add(polygons_vgroup_4a, polygons_vgroup_4b)

        # Ok dope - maybe want to mess with final layer scaling, we'll see
        # Now we just need a decision boundary. 

        # def find_polytope_intersection(polygons_1, polygons_2):
        #     '''
        #     Given two lists of Nx3 numpy arrays, (polygons_1, polygons_2), where each 
        #     numpy array gives the vertices of face of a polytope, find all intersection lines between the 
        #     two polytope surfaces, and return as a list of starting an dending points for each line. 
        #     '''

        #     return intersection_line_coords


        final_polygons=copy.deepcopy(layer_3_polygons_3d) #Need to to intersection on non-scaled polytopes
        intersection_line_coords=find_polytope_intersection(final_polygons[0], final_polygons[1])
        #Hmm like 3k results? Seems like a lot lol. Maybe analtyical in not 
        
        decision_boundary_lines = Group()
        for line_segment in intersection_line_coords:
            if len(line_segment) == 2:
                start_point, end_point = line_segment
                line = Line3D(
                    start=start_point,
                    end=end_point,
                    color="#FF00FF",
                    width=0.01,
                )
                # Shift to match your layer positioning (layer_idx=5 for final output)
                line.shift([3*5-6, 0, 0])  # horizontal_spacing * layer_idx - 6
                decision_boundary_lines.add(line)

        # Add the decision boundary to the scene
        self.add(decision_boundary_lines)

        # Ok ChatGPT's solution is getting there, it still misses a segmmetn it looks like, and 
        # I need to scale it back down for viz!
        # Ok, so I'll pick up here when I'm back and see what I can figure out!
        # Might be worth looking at what I did in my first round of sketches -> that seemed to work ok??
        # Hmm after I viz scale, will they still intersect at the right point?
        # And final final question -> intersection line Z does not seem arbitrary -> is it fixed Z somehow? 
        # If so, why, and does it help me????


        self.wait()
        self.embed()






