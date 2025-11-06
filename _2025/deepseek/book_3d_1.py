from manimlib import *
from tqdm import tqdm
from pathlib import Path


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

from manimlib.mobject.svg.svg_mobject import _convert_point_to_3d
from manimlib.logger import log

def get_attention_head(svg_path='/Users/stephen/welch_labs/deepseek/graphics/to_manim',
                       svg_file='mha_2d_segments-',
                       img_path='/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/flowchart_graphics'):

    q1=ImageMobject(str(img_path/'q_1.png'))
    q1.scale([0.0415, 0.08, 1]) 
    q1.move_to([-0.2,0.38,0]) 
    # self.add(q1)
    # self.remove(q1)      

    k1=ImageMobject(str(img_path/'k_1.png'))
    k1.scale([0.0415, 0.08, 1]) 
    k1.move_to([-0.2,-0.06,0]) 
    # self.add(k1)
    # self.remove(k1)

    v1=ImageMobject(str(img_path/'v_1.png'))
    v1.scale([0.0415, 0.08, 1]) 
    v1.move_to([-0.2,-0.48,0]) 
    # self.add(v1)
    # self.remove(v1)

    kt=ImageMobject(str(img_path/'k_1.png'))
    kt.scale([0.0215,0.035, 1])
    kt.rotate([0, 0, -PI/2]) 
    kt.move_to([0.405,0.305,0])     
    # self.add(kt)
    # self.remove(kt)

    a1=ImageMobject(str(img_path/'attention_scores.png'))
    a1.scale([0.055,0.055, 1])
    a1.move_to([0.66,0.37,0]) 
    # self.add(a1)
    # self.remove(a1)

    a2=ImageMobject(str(img_path/'attention_pattern.png'))
    a2.scale([0.13,0.13, 1])
    a2.move_to([1.27,0.25,0]) 
    # self.add(a2)
    # self.remove(a2)

    z1=ImageMobject(str(img_path/'z_1.png'))
    z1.scale([0.0425, 0.08, 1]) 
    z1.move_to([1.035,-0.48,0]) 
    # self.add(z1)
    # self.remove(z1)

    all_images=Group(q1, k1, v1, kt, a1, a2, z1)

    svg_files=list(sorted((svg_path).glob('*'+svg_file+'*')))
    # print(svg_files)

    all_labels=Group()
    for svg_file in svg_files:
        svg_image=SVGMobject(str(svg_file))
        all_labels.add(svg_image[1:]) #Thowout background
 
    large_white_connectors=SVGMobject(str(svg_path/'mha_2d_large_white_connectors_2.svg'))

    return Group(all_images, all_labels, large_white_connectors[1:])

def get_input_x(svg_path='/Users/stephen/welch_labs/deepseek/graphics/to_manim',
           svg_file='mha_2d_grouping_test.svg',
           img_path='/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/flowchart_graphics'):


    x1=ImageMobject(str(Path(img_path)/'input_1_1.png'))
    x1.scale([0.075,0.125, 1]) 
    x1.move_to([-1.45,-0.03,0])

    x2=ImageMobject(str(Path(img_path)/'input_2_1.png'))
    x2.scale([0.075,0.125, 1]) 
    x2.move_to([-1.225,-0.03,0])
    
    all_images=Group(x1, x2)
    svg_image=SVGMobject(str(Path(svg_path)/svg_file))
    x_labels_1=svg_image[1:45] 

    return Group(all_images,x_labels_1)



class poster_1(InteractiveScene):
    def construct(self):

        img_path=Path('/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek')
        svg_path=Path('/Users/stephen/welch_labs/deepseek/graphics/to_manim')

        i=0
        a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                    img_path=img_path/'gpt_2_attention_viz_4'/str(i))

        x=get_input_x(svg_path='/Users/stephen/welch_labs/deepseek/graphics/to_manim',
           svg_file='mha_2d_grouping_test.svg',
           img_path=img_path/'gpt_2_attention_viz_4')

        self.frame.reorient(0, 0, 0, (-1.39, -0.06, -0.0), 1.16)
        self.wait()
        # self.add(x[0][0])
        # self.add(x[0][1])

        self.play(FadeIn(x[0]))
        self.add(a[1][1]) #xlabels
        self.add(a[1][2]) #xlabels
        self.wait()

        self.add(a[1][16]) #deepseek dim
        self.wait()
        self.remove(a[1][16])
        self.wait()

        self.play(FadeIn(a[1][6]), self.frame.animate.reorient(0, 0, 0, (-0.65, 0.02, 0.0), 1.36), run_time=2)
        self.wait()

        # self.play(FadeIn(a[1][3]), FadeIn(a[1][4]), FadeIn(a[0][0]), FadeIn(a[0][1]))
        # self.wait()

        # self.add(a[1][3])
        # self.add(a[1][4])
        # Can i, withoug going insane, actually have these matrices be made up of rows and 
        # then break them apart and pull one out of each?

        #Ok hack on row by row version here then break into subfuncrion


        separate_row_im_path='/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/gpt_2_attention_viz_4'
        q_rows=Group()
        for row_id in range(9):
            q=ImageMobject(separate_row_im_path+'/query_row_'+str(row_id)+'.png')
            q.scale([0.0127, 0.024, 1]) 
            q.move_to([-0.2,0.492-0.028*row_id,0]) 
            q_rows.add(q)
        # self.add(q_rows)
        # self.remove(q_rows)

        k_rows=Group()
        for row_id in range(9):
            k=ImageMobject(separate_row_im_path+'/key_row_'+str(row_id)+'.png')
            k.scale([0.0127, 0.024, 1]) 
            k.move_to([-0.2,0.05-0.028*row_id,0]) 
            k_rows.add(k)
        # self.add(k_rows)
        # self.remove(k_rows)

        self.wait()
        self.play(FadeIn(a[1][3]), FadeIn(a[1][4]), FadeIn(k_rows), FadeIn(q_rows)) #Queries and Keys 
        self.wait()

        #Pan over, create space for captions, while expanding all rows
        self.play(*[q_rows[row_id].animate.shift(0.01*(8-row_id)*UP) for row_id in range(9)]+
                   [k_rows[row_id].animate.shift(0.01*(8-row_id)*UP) for row_id in range(9)]+
                   [self.frame.animate.reorient(0, 0, 0, (-0.23, 0.18, 0.0), 1.20)], lag_ratio=0.2, run_time=3)

        self.wait()

        self.play(FadeOut(x[0]), FadeOut(a[1][1]), FadeOut(a[1][2]), FadeOut(a[1][6]))
        a[1][18].scale(0.595)
        a[1][18].move_to([-0.75,0.2,0])
        self.add(a[1][18]) #Add american flag text
        self.wait()

        a[1][19].scale(0.6)
        a[1][19].move_to([0.49,0.37,0])
        self.add(a[1][19][:30]) #Key and Query Questions
        self.wait()
        self.add(a[1][19][30:])
        self.wait()

        #Multistep option
        self.play(FadeOut(a[1][18][:11]), FadeOut(a[1][18][14:32]), FadeOut(a[1][18][40:]), FadeOut(a[1][19]), 
                  FadeOut(a[1][3]), FadeOut(a[1][4]), FadeOut(q_rows[:2]), FadeOut(q_rows[3:]), FadeOut(k_rows[0]), 
                  FadeOut(k_rows[2:]), run_time=2)
        # self.wait()
        self.play(q_rows[2].animate.shift(0.12*DOWN), k_rows[1].animate.shift(0.12*UP),
                  a[1][18][11:14].animate.shift(0.12*DOWN), a[1][18][32:40].animate.shift(0.12*UP))
        self.wait()

        #All at once option - feels like to much
        # self.play(FadeOut(a[1][18][:11]), FadeOut(a[1][18][14:32]), FadeOut(a[1][18][40:]), FadeOut(a[1][19]), 
        #       FadeOut(a[1][3]), FadeOut(a[1][4]), FadeOut(q_rows[:2]), FadeOut(q_rows[3:]), FadeOut(k_rows[0]), 
        #       FadeOut(k_rows[2:]), q_rows[2].animate.shift(0.15*DOWN), k_rows[1].animate.shift(0.15*UP), run_time=2)     
        # self.wait()

        #Dot Product Dot
        d=Dot(stroke_color=None, fill_color=WHITE)
        d.scale(0.15)
        d.move_to([-0.2, 0.295, 0])
        # d.set_color(WHITE)
        self.add(d)

        t=Tex("=524.2").set_color(WHITE)
        t.scale(0.1)
        t.move_to([-0.2, 0.12, 0])
        self.add(t)
        self.wait()

        self.remove(d,t)

        # So here's just a pure reversal and everything back, but I wonder if I want to actually return to non-broken rows
        # so I can go straight into the transpose?
        # self.play(q_rows[2].animate.shift(0.12*UP), k_rows[1].animate.shift(0.12*DOWN),
        #           a[1][18][11:14].animate.shift(0.12*UP), a[1][18][32:40].animate.shift(0.12*DOWN))
        # self.wait()

        # self.add(a[1][18][:11], a[1][18][14:32], a[1][18][40:])
        # self.add(a[1][3], a[1][4])
        # self.add(q_rows[:2], q_rows[3:], k_rows[0], k_rows[2:])
        # self.wait()

        #Ok try to smoothly move back to full matrix

        # self.wait()
        self.play(q_rows[2].animate.shift(0.059*UP), k_rows[1].animate.shift(0.188*DOWN),
                  FadeOut(a[1][18][11:14]), FadeOut(a[1][18][32:40]))
        
        # self.add(a[0][0].set_opacity(0.5)) #Query image
        # q_rows[2].shift(0.059*UP)
        # self.wait()
        # self.add(a[0][1].set_opacity(0.5)) #Key image
        # k_rows[1].shift(0.188*DOWN)

        self.play(FadeIn(a[0][0]), FadeIn(a[0][1]))
        self.remove(q_rows[2], k_rows[1])
        self.add(a[1][3][:10], a[1][3][-1:]) #Queryt labels
        self.add(a[1][4][:4], a[1][4][-4:]) #Key labels
        self.wait()

        #Ok time to transpose a copy of the keys and add next set of labels and results -> how can i do this smoothy?
        # self.add(a[0][3])

        kt=a[0][1].copy()
        self.play(kt.animate.scale(np.array([0.0215,0.035, 1])/np.array([0.0415, 0.08, 1])).rotate([0, 0, -PI/2]).rotate(PI,UP).move_to([0.405,0.305,0]), run_time=2)  
        # self.play(kt.animate.rotate(PI,RIGHT).rotate([0, 0, -PI/2]))
        self.add(a[1][7])
        self.wait()
        # self.remove(kt)


        self.play(FadeIn(a[1][8]), FadeIn(a[0][4]), self.frame.animate.reorient(0, 0, 0, (0.16, 0.17, 0.0), 1.20), run_time=1.2)
        self.wait()

        self.add(a[1][9])
        self.wait()
        self.play(FadeIn(a[0][5]), self.frame.animate.reorient(0, 0, 0, (0.46, 0.18, 0.0), 1.43))
        self.wait()
        self.add(a[1][10])
        self.wait()

        #P18 - move then add option
        self.play(self.frame.animate.reorient(0, 0, 0, (0, 0, 0.0), 2.00))
        self.remove(a[1][3][:10], a[1][3][-1:]) #Remove partial labels, add back in full. 
        self.remove(a[1][4][:4], a[1][4][-4:])
        self.add(a[1][3], a[1][4]) #Key query value labels
        self.add(x[0], a[1][1], a[1][2], a[1][6]) #X inputs and intiial arrows
        self.wait()

        self.play(FadeIn(a[1][17]), FadeIn(a[1][5]), FadeIn(a[0][2])) #Value Stuff
        self.wait()

        ## Zoom in on values
        self.play(self.frame.animate.reorient(0, 0, 0, (-0.22, -0.49, -0.0), 1.15))
        self.wait()

        ## Zoom out partially to make comparison to keys
        self.play(self.frame.animate.reorient(0, 0, 0, (-0.16, -0.09, 0.0), 1.35))
        self.wait()

        ## Fully Zoom out
        self.play(self.frame.animate.reorient(0, 0, 0, (0, 0, 0.0), 2.00))
        self.wait()

        #P19
        self.add(a[1][11])
        self.wait()

        self.play(FadeIn(a[0][6]), FadeIn(a[1][12]))
        self.wait()

        #Attention head border 
        self.play(Write(a[1][0]), run_time=1) #kind cool kind cheesy, switch to fade in if I hate it 
        # self.play(FadeIn(a[1][0]))
        self.wait()

        # Ok now we're getting serious. I need to load up the other 11 heads, bring them in, and maybe like
        # pan to the side as I do it. Maybe I can do a lag ratio thing. Oh yeah shoot and my camera is 
        # top down right now, and for decent pan I need to rotate everything up. Hmm. 
        # Would it be insane to rotate the attention head up like as I did it?

        #Quick hacky test - a little weird but might work?
        #Oh man can my patters slot in from the left one after eachother as the camera moves alittle?
        #That would be dope. 
        head_0_3d_images=Group(a[0], kt) #a[0][:2], a[3:], kt)
        head_0_3d_vectors=Group(a[1][0], a[1][3:6], a[1][6:13], a[1][17])
        head_0_3d=Group(head_0_3d_images, head_0_3d_vectors)

        # Ok here's my alternative solution - there should not appear to be a camera jump in theory, we'll see - 
        # If there is a little one, by be able to fix in Premier
        self.remove(x, a[1][1], a[1][2])
        head_0_3d.rotate([PI/2,0,0], axis=RIGHT)
        self.frame.reorient(0, 90, 0, (0, 0, 0.0), 2.0)
        self.wait()

        # self.play(head_0_3d.animate.rotate([PI/2,0,0], axis=RIGHT), 
        #          self.frame.animate.reorient(0, 90, 0, (0, 0, 0.0), 2.0), run_time=2)

        #Ok now the cool stuff, can I slot in new heads from the left while panning to left?!
        attention_heads=Group()
        spacing=0.25
        for i in range(1, 12): 
            a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                        img_path=img_path/'gpt_2_attention_viz_4'/str(i))
            a_select=Group(a[0], a[1][0], a[1][3:6], a[1][6:13], a[1][17])
            a_select.rotate([PI/2,0,0], axis=RIGHT)
            a_select.move_to([0.27, spacing*i,0])
            attention_heads.add(a_select)

        # self.wait()
        # self.frame.reorient(-27, 74, 0, (-0.29, 0.14, -0.1), 2.56)

        #What if I add heads and then set all their opacities to zero? Is ordering preserved then?
        for i in range(10, -1, -1): #-1):
            # print(i)
            self.add(attention_heads[i])
        self.add(head_0_3d)

        for i in range(10, -1, -1): 
            attention_heads[i].shift(6*LEFT)

        # self.wait()
        # self.frame.reorient(-38, 72, 0, (-0.37, 0.29, 0.02), 2.56)
        self.play(self.frame.animate.reorient(-38, 72, 0, (-0.37, 0.29, 0.02), 2.56), run_time=3)

        # self.play(attention_heads[0].animate.shift(6*RIGHT)) #Works
        # self.play(attention_heads[1].animate.shift(6*RIGHT)) #Works

        self.wait(0)
        for i in range(11): #not my favorite, but gets the job done. 
            self.play(attention_heads[i].animate.shift(6*RIGHT), run_time=1, rate_func=linear)

        self.wait()

        self.play(self.frame.animate.reorient(36, 83, 0, (0.8, 0.62, -0.11), 2.65), run_time=6)
        self.wait()

        #P20 -> pull out output matrices
        out_matrices=[attention_heads[i][0][-1] for i in range(10, -1, -1)] + [head_0_3d[0][0][-1]]

        # for i,o in enumerate(out_matrices):
        #     o.scale(0.2).move_to([3+i*0.2, 0, 0])

        self.play(*[o.animate.scale(0.2).move_to([3+i*0.2, 0, 0]) for i,o in enumerate(out_matrices)]+
                  [self.frame.animate.reorient(-1, 86, 0, (4.32, 0.84, -0.21), 3.29)], run_time=3)
        self.wait()

        p20_overlay=SVGMobject('/Users/stephen/welch_labs/deepseek/graphics/to_manim/p20_svg_overay_1.svg')[1:]
        p20_overlay.rotate([PI/2,0,0], axis=RIGHT)
        p20_overlay.scale(1.3)
        p20_overlay.move_to([4.62,0,-0.26])
        self.add(p20_overlay)

        # self.remove(p20_overlay)


        # for i in range(10, -1, -1):
            # out_matrix=attention_heads[i][0][-1]
            # out_matrix.scale(0.2).move_to([2.25+i*0.25, 0, 0])
        # head_0_3d[0][0][-1].scale(0.25).move_to([2.25+(i-1)*0.25, 0, 0])




        # self.play(*[attention_heads[i].animate.shift(6*RIGHT) for i in range(11)], lag_ratio=1.2) #Breaks things down wrong 

        # animations = [attention_heads[i].animate.shift(6*RIGHT) for i in range(11)]
        # staggered_animations = AnimationGroup(*animations, lag_ratio=1.5)
        # self.play(staggered_animations, run_time=2)


        # self.play(attention_heads[i].animate.shift(6*RIGHT))
        # self.frame.reorient(-27, 74, 0, (-0.29, 0.14, -0.1), 2.56)


        # attention_heads.set_opacity(0.0) #Ok ordering seems to survive this simple test
        # attention_heads.set_opacity(1.0)

        #Ok seems like the animation group was actually fucking me in terms of occlusions???
        # animations = [attention_heads[i].animate.set_opacity(0.85) for i in range(11)]
        # staggered_animations = AnimationGroup(*animations, lag_ratio=1.5)
        # self.play(staggered_animations, run_time=2)

        #Well shit I may just have to call this one a loss for now and try later if I have time
        #The behavior I can't seem to create is bringin in the heads on at a time, front to back, without fucking occlusions. 


        #Hmm I'm going to possibly run into some rendering order stuff? 
        #maybe not if I render them off screen in the right order and then bring them in?
        # for i in range(10, -1, -1):
        # for i in range(10,-1,-1):
        #     attention_heads[i].set_opacity(0.0)

        #     # attention_heads[i].shift(6*LEFT) #Not sure what starting position is?
        #     self.add(attention_heads[i])
        # Set z_index based on depth - higher values render on top
        # for i in range(11):
        #     # Calculate z_index based on depth (negative z_coordinate)
        #     z_coord = attention_heads[i].get_center()[2]
        #     # Convert z coordinate to z_index - smaller z should have higher z_index to be on top
        #     z_index = -int(z_coord * 100)  # Scale appropriately for your scene
        #     attention_heads[i].set_z_index(z_index)
        #     attention_heads[i].set_opacity(1.0)
        #     self.add(attention_heads[i])

        # self.add(head_0_3d) #Re add as top layer
        # self.wait()

        # attention_heads.shift(0.25*RIGHT)

        # self.frame.reorient(-27, 74, 0, (-0.29, 0.14, -0.1), 2.56)

        # # animations = [attention_heads[i].animate.shift(6*RIGHT) for i in range(11)]
        # animations = [FadeIn(attention_heads[i]) for i in range(11)]
        # staggered_animations = AnimationGroup(*animations, lag_ratio=1.5)
        # self.play(staggered_animations, run_time=2)

        # head_0_3d.set_z_index(100) #Doesn't seem eo work. 
        # self.add(head_0_3d)

        #Man this is tricky! No matter how I introduce the additional layers, I can't seem to get the occlusion 
        #ordering right, even with seeing z order manually. Am I missing something here?



        # self.play(*[attention_heads[i].animate.shift(6*RIGHT) for i in range(11)], run_time=3, lag_ratio=1.5)


        # self.wait()






        self.wait(20)
        self.embed()


class poster_1_b(InteractiveScene):
    '''BUnch of extra stuff in here but meh'''
    def construct(self):

        img_path=Path('/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek')
        svg_path=Path('/Users/stephen/welch_labs/deepseek/graphics/to_manim')

        attention_heads=Group()
        spacing=0.25
        for i in range(12): 
            a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                        img_path=img_path/'gpt_2_attention_viz_4'/str(i))
            # a_select=Group(a[0], a[1][0], a[1][3:6], a[1][6:13], a[1][17])
            # a_select=Group(a[0], a[1][3:7], a[1][13], a[1][17])
            a.rotate([PI/2,0,0], axis=RIGHT)
            a.move_to([0.27, spacing*i,0])
            attention_heads.add(a)

        for i in range(11, 0, -1): #-1):
            self.add(attention_heads[i][0])
            # self.add(attention_heads[i][1][0]) 
            self.add(attention_heads[i][1][20])
            self.add(attention_heads[i][1][13]) 
            self.add(attention_heads[i][1][21])

        # attention_heads.set_opacity(0.5)
        # attention_heads[0].set_opacity(1.0)

        i=0
        # self.add(attention_heads[i][0])
        self.add(attention_heads[i][0]) 
        # self.add(attention_heads[i][1][0]) 
        self.add(attention_heads[i][1][3:6])
        self.add(attention_heads[i][1][6:13]) 
        self.add(attention_heads[i][1][17])



        ### --- Get poster fram here.  ---- ###
        # self.frame.reorient(51, 74, 0, (0.05, 0.82, -0.29), 3.44)
        # self.frame.reorient(-38, 59, 0, (0.12, 0.93, -0.18), 2.93)

        #Book bro
        self.frame.reorient(32, 84, 0, (0.57, 1.22, -0.24), 3.40)
        self.wait()

        self.frame.reorient(33, 85, 0, (0.71, 1.32, -0.14), 2.88)
        self.wait()


        self.wait(20)
        self.embed()





class book_3d_1b(InteractiveScene):
    '''BUnch of extra stuff in here but meh'''
    def construct(self):

        img_path=Path('/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek')
        svg_path=Path('/Users/stephen/welch_labs/deepseek/graphics/to_manim')

        i=0
        a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                    img_path=img_path/'gpt_2_attention_viz_4'/str(i))

        x=get_input_x(svg_path='/Users/stephen/welch_labs/deepseek/graphics/to_manim',
           svg_file='mha_2d_grouping_test.svg',
           img_path='/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/flowchart_graphics')

        self.frame.reorient(0, 0, 0, (-1.39, -0.06, -0.0), 1.16)
        self.wait()
        # self.add(x[0][0])
        # self.add(x[0][1])

        self.play(FadeIn(x[0]))
        self.add(a[1][1]) #xlabels
        self.add(a[1][2]) #xlabels
        self.wait()

        self.add(a[1][16]) #deepseek dim
        self.wait()
        self.remove(a[1][16])
        self.wait()

        self.play(FadeIn(a[1][6]), self.frame.animate.reorient(0, 0, 0, (-0.65, 0.02, 0.0), 1.36), run_time=2)
        self.wait()

        # self.play(FadeIn(a[1][3]), FadeIn(a[1][4]), FadeIn(a[0][0]), FadeIn(a[0][1]))
        # self.wait()

        # self.add(a[1][3])
        # self.add(a[1][4])
        # Can i, withoug going insane, actually have these matrices be made up of rows and 
        # then break them apart and pull one out of each?

        #Ok hack on row by row version here then break into subfuncrion


        separate_row_im_path='/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/gpt_2_attention_viz_4'
        q_rows=Group()
        for row_id in range(9):
            q=ImageMobject(separate_row_im_path+'/query_row_'+str(row_id)+'.png')
            q.scale([0.0127, 0.024, 1]) 
            q.move_to([-0.2,0.492-0.028*row_id,0]) 
            q_rows.add(q)
        # self.add(q_rows)
        # self.remove(q_rows)

        k_rows=Group()
        for row_id in range(9):
            k=ImageMobject(separate_row_im_path+'/key_row_'+str(row_id)+'.png')
            k.scale([0.0127, 0.024, 1]) 
            k.move_to([-0.2,0.05-0.028*row_id,0]) 
            k_rows.add(k)
        # self.add(k_rows)
        # self.remove(k_rows)

        self.wait()
        self.play(FadeIn(a[1][3]), FadeIn(a[1][4]), FadeIn(k_rows), FadeIn(q_rows)) #Queries and Keys 
        self.wait()

        #Pan over, create space for captions, while expanding all rows
        self.play(*[q_rows[row_id].animate.shift(0.01*(8-row_id)*UP) for row_id in range(9)]+
                   [k_rows[row_id].animate.shift(0.01*(8-row_id)*UP) for row_id in range(9)]+
                   [self.frame.animate.reorient(0, 0, 0, (-0.23, 0.18, 0.0), 1.20)], lag_ratio=0.2, run_time=3)

        self.wait()

        self.play(FadeOut(x[0]), FadeOut(a[1][1]), FadeOut(a[1][2]), FadeOut(a[1][6]))
        a[1][18].scale(0.595)
        a[1][18].move_to([-0.75,0.2,0])
        self.add(a[1][18]) #Add american flag text
        self.wait()

        a[1][19].scale(0.6)
        a[1][19].move_to([0.49,0.37,0])
        self.add(a[1][19][:30]) #Key and Query Questions
        self.wait()
        self.add(a[1][19][30:])
        self.wait()

        #Multistep option
        self.play(FadeOut(a[1][18][:11]), FadeOut(a[1][18][14:32]), FadeOut(a[1][18][40:]), FadeOut(a[1][19]), 
                  FadeOut(a[1][3]), FadeOut(a[1][4]), FadeOut(q_rows[:2]), FadeOut(q_rows[3:]), FadeOut(k_rows[0]), 
                  FadeOut(k_rows[2:]), run_time=2)
        # self.wait()
        self.play(q_rows[2].animate.shift(0.12*DOWN), k_rows[1].animate.shift(0.12*UP),
                  a[1][18][11:14].animate.shift(0.12*DOWN), a[1][18][32:40].animate.shift(0.12*UP))
        self.wait()

        #All at once option - feels like to much
        # self.play(FadeOut(a[1][18][:11]), FadeOut(a[1][18][14:32]), FadeOut(a[1][18][40:]), FadeOut(a[1][19]), 
        #       FadeOut(a[1][3]), FadeOut(a[1][4]), FadeOut(q_rows[:2]), FadeOut(q_rows[3:]), FadeOut(k_rows[0]), 
        #       FadeOut(k_rows[2:]), q_rows[2].animate.shift(0.15*DOWN), k_rows[1].animate.shift(0.15*UP), run_time=2)     
        # self.wait()

        #Dot Product Dot
        d=Dot(stroke_color=None, fill_color=WHITE)
        d.scale(0.15)
        d.move_to([-0.2, 0.295, 0])
        # d.set_color(WHITE)
        self.add(d)

        t=Tex("=524.2").set_color(WHITE)
        t.scale(0.1)
        t.move_to([-0.2, 0.12, 0])
        self.add(t)
        self.wait()

        self.remove(d,t)

        # So here's just a pure reversal and everything back, but I wonder if I want to actually return to non-broken rows
        # so I can go straight into the transpose?
        # self.play(q_rows[2].animate.shift(0.12*UP), k_rows[1].animate.shift(0.12*DOWN),
        #           a[1][18][11:14].animate.shift(0.12*UP), a[1][18][32:40].animate.shift(0.12*DOWN))
        # self.wait()

        # self.add(a[1][18][:11], a[1][18][14:32], a[1][18][40:])
        # self.add(a[1][3], a[1][4])
        # self.add(q_rows[:2], q_rows[3:], k_rows[0], k_rows[2:])
        # self.wait()

        #Ok try to smoothly move back to full matrix

        # self.wait()
        self.play(q_rows[2].animate.shift(0.059*UP), k_rows[1].animate.shift(0.188*DOWN),
                  FadeOut(a[1][18][11:14]), FadeOut(a[1][18][32:40]))
        
        # self.add(a[0][0].set_opacity(0.5)) #Query image
        # q_rows[2].shift(0.059*UP)
        # self.wait()
        # self.add(a[0][1].set_opacity(0.5)) #Key image
        # k_rows[1].shift(0.188*DOWN)

        self.play(FadeIn(a[0][0]), FadeIn(a[0][1]))
        self.remove(q_rows[2], k_rows[1])
        self.add(a[1][3][:10], a[1][3][-1:]) #Queryt labels
        self.add(a[1][4][:4], a[1][4][-4:]) #Key labels
        self.wait()

        #Ok time to transpose a copy of the keys and add next set of labels and results -> how can i do this smoothy?
        # self.add(a[0][3])

        kt=a[0][1].copy()
        self.play(kt.animate.scale(np.array([0.0215,0.035, 1])/np.array([0.0415, 0.08, 1])).rotate([0, 0, -PI/2]).rotate(PI,UP).move_to([0.405,0.305,0]), run_time=2)  
        self.add(a[1][7])
        self.wait()


        self.play(FadeIn(a[1][8]), FadeIn(a[0][4]), self.frame.animate.reorient(0, 0, 0, (0.16, 0.17, 0.0), 1.20), run_time=1.2)
        self.wait()

        self.add(a[1][9])
        self.wait()
        self.play(FadeIn(a[0][5]), self.frame.animate.reorient(0, 0, 0, (0.46, 0.18, 0.0), 1.43))
        self.wait()
        self.add(a[1][10])
        self.wait()

        #P18 - move then add option
        self.play(self.frame.animate.reorient(0, 0, 0, (0, 0, 0.0), 2.00))
        self.remove(a[1][3][:10], a[1][3][-1:]) #Remove partial labels, add back in full. 
        self.remove(a[1][4][:4], a[1][4][-4:])
        self.add(a[1][3], a[1][4]) #Key query value labels
        self.add(x[0], a[1][1], a[1][2], a[1][6]) #X inputs and intiial arrows
        self.wait()

        self.play(FadeIn(a[1][17]), FadeIn(a[1][5]), FadeIn(a[0][2])) #Value Stuff
        self.wait()

        #P19
        self.add(a[1][11])
        self.wait()

        self.play(FadeIn(a[0][6]), FadeIn(a[1][12]))
        self.wait()

        #Attention head border 
        # self.play(Write(a[1][0]), run_time=1) #kind cool kind cheesy, switch to fade in if I hate it 
        # self.play(FadeIn(a[1][0]))
        self.wait()

        # Ok now we're getting serious. I need to load up the other 11 heads, bring them in, and maybe like
        # pan to the side as I do it. Maybe I can do a lag ratio thing. Oh yeah shoot and my camera is 
        # top down right now, and for decent pan I need to rotate everything up. Hmm. 
        # Would it be insane to rotate the attention head up like as I did it?

        #Quick hacky test - a little weird but might work?
        #Oh man can my patters slot in from the left one after eachother as the camera moves alittle?
        #That would be dope. 
        head_0_3d_images=Group(a[0], kt) #a[0][:2], a[3:], kt)
        # head_0_3d_vectors=Group(a[1][0], a[1][3:6], a[1][6:13], a[1][17])
        head_0_3d_vectors=Group(a[1][3:6], a[1][6:13], a[1][17])
        head_0_3d=Group(head_0_3d_images, head_0_3d_vectors)

        # Ok here's my alternative solution - there should not appear to be a camera jump in theory, we'll see - 
        # If there is a little one, by be able to fix in Premier
        self.remove(x, a[1][1], a[1][2])
        head_0_3d.rotate([PI/2,0,0], axis=RIGHT)
        self.frame.reorient(0, 90, 0, (0, 0, 0.0), 2.0)
        self.wait()

        # self.play(head_0_3d.animate.rotate([PI/2,0,0], axis=RIGHT), 
        #          self.frame.animate.reorient(0, 90, 0, (0, 0, 0.0), 2.0), run_time=2)

        #Ok now the cool stuff, can I slot in new heads from the left while panning to left?!
        attention_heads=Group()
        spacing=0.25
        for i in range(1, 12): 
            a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                        img_path=img_path/'gpt_2_attention_viz_4'/str(i))
            # a_select=Group(a[0], a[1][0], a[1][3:6], a[1][6:13], a[1][17])
            a_select=Group(a[0], a[1][3:7], a[1][13], a[1][17])
            a_select.rotate([PI/2,0,0], axis=RIGHT)
            a_select.move_to([0.27, spacing*i,0])
            attention_heads.add(a_select)

        # self.wait()
        # self.frame.reorient(-27, 74, 0, (-0.29, 0.14, -0.1), 2.56)

        #What if I add heads and then set all their opacities to zero? Is ordering preserved then?
        for i in range(10, -1, -1): #-1):
            # print(i)
            self.add(attention_heads[i])
        self.add(head_0_3d)

        for i in range(10, -1, -1): 
            attention_heads[i].shift(6*LEFT)

        self.wait()
        # self.frame.reorient(-38, 72, 0, (-0.37, 0.29, 0.02), 2.56)
        self.play(self.frame.animate.reorient(-38, 72, 0, (-0.37, 0.29, 0.02), 2.56), run_time=3)


        for i in range(11): #not my favorite, but gets the job done. 
            self.play(attention_heads[i].animate.shift(6*RIGHT), run_time=1, rate_func=linear)

        self.wait()

        ### --- Get poster fram here.  ---- ###
        self.frame.reorient(51, 74, 0, (0.05, 0.82, -0.29), 3.44)



        # self.play(self.frame.animate.reorient(-48, 70, 0, (0.05, 0.82, -0.29), 3.44), run_time=10, rate_func=linear)
        self.wait()

        #Second book frame grab
        self.wait()
        self.frame.reorient(-33, 69, 0, (0.26, 0.84, -0.18), 2.42)
        self.wait()
        self.frame.reorient(-38, 66, 0, (0.28, 0.84, -0.15), 2.31)
        self.wait()
        # self.frame.reorient(-38, 62, 0, (0.17, 0.82, -0.32), 2.69)
        # self.frame.reorient(-38, 62, 0, (0.21, 0.8, -0.31), 2.31)
        self.frame.reorient(-47, 62, 0, (0.06, 0.97, -0.28), 2.87)
        self.wait()
        

        self.wait(20)
        self.embed()




        # q1=ImageMobject(str(img_path/'q_1.png'))
        # q1.scale([0.0415, 0.08, 1]) 
        # q1.move_to([-0.2,0.38,0]) 
        # # self.add(q1)
        # # self.remove(q1)      

        # k1=ImageMobject(str(img_path/'k_1.png'))
        # k1.scale([0.0415, 0.08, 1]) 
        # k1.move_to([-0.2,-0.06,0]) 



        # self.wait()
        # self.embed()

        #         attention_heads=Group()
        # spacing=0.25
        # for i in range(12): #Render in reverse order for occlusions
        #     a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
        #                                 img_path=img_path/'gpt_2_attention_viz_1'/str(i))
        #     a.rotate([PI/2,0,0], axis=RIGHT)
        #     a.move_to([0, spacing*i,0])
        #     # a.set_opacity(0.5)
        #     attention_heads.add(a)


        # a=get_attention_head_course(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1/0', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(a)

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(x)

        # self.frame.reorient(0, 0, 0, (0.00, 0.00, 0.00), 2.80)
        # # self.wait()



