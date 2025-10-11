from manimlib import *from manimlib import *

import numpy as npfrom functools import partial

import matplotlib.colors as mcolorsimport numpy as np

import matplotlib.pyplot as pltimport sys

import matplotlib.colors as mcolors

CHILL_BROWN = "#948979"import matplotlib.pyplot as plt

FRESH_TAN = "#efe5d1"import glob

YELLOW = "#ffd35a"

BLUE = "#65c8d0"CHILL_BROWN = "#948979"

GREEN = "#00a14b"FRESH_TAN = "#efe5d1"

THUNDER = "#1b1619"YELLOW = "#ffd35a"

BLUE = "#65c8d0"

ASSET_PATH = "~/Stephencwelch Dropbox/welch_labs/double_descent/hackin"GREEN = "#00a14b"

THUNDER = "#1b1619"

def get_edge_points(circle1, circle2, neuron_radius):

    direction = circle2.get_center() - circle1.get_center()ASSET_PATH = "~/Stephencwelch Dropbox/welch_labs/double_descent/hackin"

    unit_vector = direction / np.linalg.norm(direction)

    start_point = circle1.get_center() + unit_vector * neuron_radius

    end_point = circle2.get_center() - unit_vector * neuron_radiusdef get_edge_points(circle1, circle2, neuron_radius):

    return start_point, end_point    direction = circle2.get_center() - circle1.get_center()

    unit_vector = direction / np.linalg.norm(direction)

viridis_colormap = plt.get_cmap("viridis")

blues_colormap = plt.get_cmap("Blues")    start_point = circle1.get_center() + unit_vector * neuron_radius

custom_cmap_tans = mcolors.LinearSegmentedColormap.from_list("custom", ["#000000", "#dfd0b9"], N=256)    end_point = circle2.get_center() - unit_vector * neuron_radius

custom_cmap_cyan = mcolors.LinearSegmentedColormap.from_list("custom", ["#000000", "#00FFFF"], N=256)

    return start_point, end_point

def get_neuron_color(value, vmax=0.95):

    value_clipped = np.clip(np.abs(value) / vmax, 0, 1)

    rgba = custom_cmap_tans(value_clipped)viridis_colormap = plt.get_cmap("viridis")

    return Color(rgb=rgba[:3])blues_colormap = plt.get_cmap("Blues")

custom_cmap_tans = mcolors.LinearSegmentedColormap.from_list(

def get_grad_color(value):    "custom", ["#000000", "#dfd0b9"], N=256

    value_clipped = np.clip(np.abs(value), 0, 1))

    rgba = custom_cmap_cyan(value_clipped)custom_cmap_cyan = mcolors.LinearSegmentedColormap.from_list(

    return Color(rgb=rgba[:3])    "custom", ["#000000", "#00FFFF"], N=256

)

def line_circle_intersection(line_start, line_end, circle_center, radius):

    direction = line_end - line_start

    to_center = line_start - circle_centerdef get_neuron_color(value, vmax=0.95):

    a = np.dot(direction, direction)    value_clipped = np.clip(np.abs(value) / vmax, 0, 1)

    b = 2 * np.dot(to_center, direction)    rgba = custom_cmap_tans(value_clipped)

    c = np.dot(to_center, to_center) - radius * radius    return Color(rgb=rgba[:3])

    discriminant = b * b - 4 * a * c

    if discriminant < 0:

        return []def get_grad_color(value):

    discriminant_sqrt = np.sqrt(discriminant)    value_clipped = np.clip(np.abs(value), 0, 1)

    t1 = (-b - discriminant_sqrt) / (2 * a)    rgba = custom_cmap_cyan(value_clipped)

    t2 = (-b + discriminant_sqrt) / (2 * a)    return Color(rgb=rgba[:3])

    intersections = []

    if 0 < t1 < 1:

        intersections.append(t1)def line_circle_intersection(line_start, line_end, circle_center, radius):

    if 0 < t2 < 1:    """

        intersections.append(t2)    Check if a line segment intersects with a circle and return intersection points.

    return sorted(intersections)    Returns a list of t values (0 to 1) where intersections occur along the line.

    """

def create_split_line(neuron1, neuron2, all_neurons, neuron_radius, exclude_neurons=None):    direction = line_end - line_start

    if exclude_neurons is None:    to_center = line_start - circle_center

        exclude_neurons = set()

    center1 = neuron1.get_center()    a = np.dot(direction, direction)

    center2 = neuron2.get_center()    b = 2 * np.dot(to_center, direction)

    direction = center2 - center1    c = np.dot(to_center, to_center) - radius * radius

    unit_vector = direction / np.linalg.norm(direction)

    full_start = center1 + unit_vector * neuron_radius    discriminant = b * b - 4 * a * c

    full_end = center2 - unit_vector * neuron_radius

    t_values = [0.0]    if discriminant < 0:

    for neuron in all_neurons:        return []

        if neuron in exclude_neurons:

            continue    discriminant_sqrt = np.sqrt(discriminant)

        intersections = line_circle_intersection(full_start, full_end, neuron.get_center(), neuron_radius)    t1 = (-b - discriminant_sqrt) / (2 * a)

        t_values.extend(intersections)    t2 = (-b + discriminant_sqrt) / (2 * a)

    t_values.append(1.0)

    t_values = sorted(set(t_values))    intersections = []

    line_segments = VGroup()    if 0 < t1 < 1:

    for i in range(len(t_values) - 1):        intersections.append(t1)

        t_start = t_values[i]    if 0 < t2 < 1:

        t_end = t_values[i + 1]        intersections.append(t2)

        t_mid = (t_start + t_end) / 2

        mid_point = full_start + t_mid * (full_end - full_start)    return sorted(intersections)

        is_inside_neuron = False

        for neuron in all_neurons:

            if neuron in exclude_neurons:def create_split_line(neuron1, neuron2, all_neurons, neuron_radius, exclude_neurons=None):

                continue    """

            dist = np.linalg.norm(mid_point - neuron.get_center())    Create a line that splits around neurons it passes through.

            if dist < neuron_radius * 0.95:    Returns a VGroup of line segments that avoid neurons.

                is_inside_neuron = True    The line segments touch the edges of the source and destination neurons.

                break    """

        if not is_inside_neuron:    if exclude_neurons is None:

            seg_start = full_start + t_start * (full_end - full_start)        exclude_neurons = set()

            seg_end = full_start + t_end * (full_end - full_start)

            segment = Line(seg_start, seg_end)    center1 = neuron1.get_center()

            line_segments.add(segment)    center2 = neuron2.get_center()

    return line_segments    

    direction = center2 - center1

class AttentionPattern(VMobject):    unit_vector = direction / np.linalg.norm(direction)

    def __init__(self, matrix, square_size=0.3, min_opacity=0.0, max_opacity=1.0, stroke_width=1.0, viz_scaling_factor=2.5, stroke_color=FRESH_TAN, colormap=custom_cmap_tans, **kwargs):    

        super().__init__(**kwargs)    full_start = center1 + unit_vector * neuron_radius

        self.matrix = np.array(matrix)    full_end = center2 - unit_vector * neuron_radius

        self.n_rows, self.n_cols = self.matrix.shape

        self.square_size = square_size    t_values = [0.0]

        self.min_opacity = min_opacity

        self.max_opacity = np.max(self.matrix)    for neuron in all_neurons:

        self.stroke_width = stroke_width        if neuron in exclude_neurons:

        self.stroke_color = stroke_color            continue

        self._colormap = colormap

        self.viz_scaling_factor = viz_scaling_factor        intersections = line_circle_intersection(

        self.build()            full_start, full_end, neuron.get_center(), neuron_radius

        )

    def map_value_to_style(self, val):        t_values.extend(intersections)

        val_scaled = np.clip(self.viz_scaling_factor * val / self.max_opacity, 0, 1)

        rgba = self._colormap(val_scaled)    t_values.append(1.0)

        color = Color(rgb=rgba[:3])    t_values = sorted(set(t_values))

        opacity = 1.0

        return {"color": color, "opacity": opacity}    line_segments = VGroup()

    for i in range(len(t_values) - 1):

    def build(self):        t_start = t_values[i]

        self.clear()        t_end = t_values[i + 1]

        squares = VGroup()        t_mid = (t_start + t_end) / 2

        for i in range(self.n_rows):

            for j in range(self.n_cols):        mid_point = full_start + t_mid * (full_end - full_start)

                val = self.matrix[i, j]

                style = self.map_value_to_style(val)        is_inside_neuron = False

                square = Square(side_length=self.square_size)        for neuron in all_neurons:

                square.set_fill(style["color"], opacity=style["opacity"])            if neuron in exclude_neurons:

                square.set_stroke(self.stroke_color, width=self.stroke_width)                continue

                pos = RIGHT * j * self.square_size + DOWN * i * self.square_size            dist = np.linalg.norm(mid_point - neuron.get_center())

                square.move_to(pos)            if dist < neuron_radius * 0.95:

                squares.add(square)                is_inside_neuron = True

        squares.move_to(ORIGIN)                break

        self.add(squares)

        if not is_inside_neuron:

def get_mlp(w1, w2, neuron_fills=None, grads_1=None, grads_2=None, line_weight=1.0, line_opacity=0.5, neuron_stroke_width=1.0, neuron_stroke_color="#dfd0b9", line_stroke_color="#948979", connection_display_thresh=0.4):            seg_start = full_start + t_start * (full_end - full_start)

    INPUT_NEURONS = w1.shape[0]            seg_end = full_start + t_end * (full_end - full_start)

    HIDDEN_NEURONS = w1.shape[1]            segment = Line(seg_start, seg_end)

    OUTPUT_NEURONS = w1.shape[0]            line_segments.add(segment)

    NEURON_RADIUS = 0.06

    LAYER_SPACING = 0.23    return line_segments

    VERTICAL_SPACING = 0.18

    DOTS_SCALE = 0.5

    input_layer = VGroup()class AttentionPattern(VMobject):

    hidden_layer = VGroup()    def __init__(

    output_layer = VGroup()        self,

    dots = VGroup()        matrix,

    neuron_count = 0        square_size=0.3,

    for i in range(INPUT_NEURONS):        min_opacity=0.0,

        if i == w1.shape[0] // 2:        max_opacity=1.0,

            dot = Tex("...").rotate(PI / 2, OUT).scale(DOTS_SCALE).move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS // 2 - i) * VERTICAL_SPACING))        stroke_width=1.0,

            dot.set_color(neuron_stroke_color)        viz_scaling_factor=2.5,

            dots.add(dot)        stroke_color=FRESH_TAN,

        else:        colormap=custom_cmap_tans,

            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)        **kwargs

            neuron.set_stroke(width=neuron_stroke_width)    ):

            if neuron_fills is None:        super().__init__(**kwargs)

                neuron.set_fill(color="#000000", opacity=1.0)        self.matrix = np.array(matrix)

            else:        self.n_rows, self.n_cols = self.matrix.shape

                neuron.set_fill(color=get_neuron_color(neuron_fills[0][neuron_count], vmax=np.abs(neuron_fills[0]).max()), opacity=1.0)        self.square_size = square_size

            neuron.move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS // 2 - i) * VERTICAL_SPACING))        self.min_opacity = min_opacity

            input_layer.add(neuron)        self.max_opacity = np.max(self.matrix)

            neuron_count += 1        self.stroke_width = stroke_width

    neuron_count = 0        self.stroke_color = stroke_color

    for i in range(HIDDEN_NEURONS):        self._colormap = colormap

        if i == w1.shape[1] // 2:        self.viz_scaling_factor = viz_scaling_factor

            dot = Tex("...").rotate(PI / 2, OUT).scale(DOTS_SCALE).move_to(UP * ((HIDDEN_NEURONS // 2 - i) * VERTICAL_SPACING))

            dot.set_color(neuron_stroke_color)        self.build()

            dots.add(dot)

        else:    def map_value_to_style(self, val):

            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)        val_scaled = np.clip(self.viz_scaling_factor * val / self.max_opacity, 0, 1)

            neuron.set_stroke(width=neuron_stroke_width)        rgba = self._colormap(val_scaled)

            if neuron_fills is None:        color = Color(rgb=rgba[:3])

                neuron.set_fill(color="#000000", opacity=1.0)        opacity = 1.0

            else:        return {"color": color, "opacity": opacity}

                neuron.set_fill(color=get_neuron_color(neuron_fills[1][neuron_count], vmax=np.abs(neuron_fills[1]).max()), opacity=1.0)

            neuron.move_to(UP * ((HIDDEN_NEURONS // 2 - i) * VERTICAL_SPACING))    def build(self):

            hidden_layer.add(neuron)        self.clear()

            neuron_count += 1        squares = VGroup()

    neuron_count = 0        for i in range(self.n_rows):

    for i in range(OUTPUT_NEURONS):            for j in range(self.n_cols):

        if i == w1.shape[0] // 2:                val = self.matrix[i, j]

            dot = Tex("...").rotate(PI / 2, OUT).scale(DOTS_SCALE).move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS // 2 - i) * VERTICAL_SPACING))                style = self.map_value_to_style(val)

            dot.set_color(neuron_stroke_color)

            dots.add(dot)                square = Square(side_length=self.square_size)

        else:                square.set_fill(style["color"], opacity=style["opacity"])

            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)                square.set_stroke(self.stroke_color, width=self.stroke_width)

            neuron.set_stroke(width=neuron_stroke_width)

            if neuron_fills is None:                pos = RIGHT * j * self.square_size + DOWN * i * self.square_size

                neuron.set_fill(color="#000000", opacity=1.0)                square.move_to(pos)

            else:                squares.add(square)

                neuron.set_fill(color=get_neuron_color(neuron_fills[2][neuron_count], vmax=np.abs(neuron_fills[2]).max()), opacity=1.0)

            neuron.move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS // 2 - i) * VERTICAL_SPACING))        squares.move_to(ORIGIN)

            output_layer.add(neuron)        self.add(squares)

            neuron_count += 1

    connections = VGroup()

    w1_abs = np.abs(w1)def get_mlp(

    w1_scaled = w1_abs / np.percentile(w1_abs, 99)    w1,

    for i, in_neuron in enumerate(input_layer):    w2,

        for j, hidden_neuron in enumerate(hidden_layer):    neuron_fills=None,

            if np.abs(w1_scaled[i, j]) < 0.75:    grads_1=None,

                continue    grads_2=None,

            if abs(i - j) > 6:    line_weight=1.0,

                continue    line_opacity=0.5,

            start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)    neuron_stroke_width=1.0,

            line = Line(start_point, end_point)    neuron_stroke_color="#dfd0b9",

            line.set_stroke(opacity=np.clip(w1_scaled[i, j], 0, 1), width=1.0 * w1_scaled[i, j])    line_stroke_color="#948979",

            line.set_color(line_stroke_color)    connection_display_thresh=0.4,

            connections.add(line)):

    w2_abs = np.abs(w2)

    w2_scaled = w2_abs / np.percentile(w2_abs, 99)    INPUT_NEURONS = w1.shape[0]

    for i, hidden_neuron in enumerate(hidden_layer):    HIDDEN_NEURONS = w1.shape[1]

        for j, out_neuron in enumerate(output_layer):    OUTPUT_NEURONS = w1.shape[0]

            if np.abs(w2_scaled[i, j]) < 0.45:    NEURON_RADIUS = 0.06

                continue    LAYER_SPACING = 0.23

            if abs(i - j) > 6:    VERTICAL_SPACING = 0.18

                continue    DOTS_SCALE = 0.5

            start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)

            line = Line(start_point, end_point)    input_layer = VGroup()

            line.set_stroke(opacity=np.clip(w2_scaled[i, j], 0, 1), width=1.0 * w2_scaled[i, j])    hidden_layer = VGroup()

            line.set_color(line_stroke_color)    output_layer = VGroup()

            connections.add(line)    dots = VGroup()

    grad_conections = VGroup()

    if grads_1 is not None:    neuron_count = 0

        grads_1_abs = np.abs(grads_1)    for i in range(INPUT_NEURONS):

        grads_1_scaled = grads_1_abs / np.percentile(grads_1_abs, 95)        if i == w1.shape[0] // 2:

        for i, in_neuron in enumerate(input_layer):            dot = (

            for j, hidden_neuron in enumerate(hidden_layer):                Tex("...")

                if np.abs(grads_1_scaled[i, j]) < 0.5:                .rotate(PI / 2, OUT)

                    continue                .scale(DOTS_SCALE)

                if abs(i - j) > 6:                .move_to(

                    continue                    LEFT * LAYER_SPACING

                start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)                    + UP * ((INPUT_NEURONS // 2 - i) * VERTICAL_SPACING)

                line_grad = Line(start_point, end_point)                )

                line_grad.set_stroke(opacity=np.clip(grads_1_scaled[i, j], 0, 1), width=np.clip(2.0 * grads_1_scaled[i, j], 0, 3))            )

                line_grad.set_color(get_grad_color(grads_1_scaled[i, j]))            dot.set_color(neuron_stroke_color)

                grad_conections.add(line_grad)            dots.add(dot)

    if grads_2 is not None:        else:

        grads_2_abs = np.abs(grads_2)            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)

        grads_2_scaled = grads_2_abs / np.percentile(grads_2_abs, 97)            neuron.set_stroke(width=neuron_stroke_width)

        for i, hidden_neuron in enumerate(hidden_layer):            if neuron_fills is None:

            for j, out_neuron in enumerate(output_layer):                neuron.set_fill(color="#000000", opacity=1.0)

                if np.abs(grads_2_scaled[i, j]) < 0.5:            else:

                    continue                neuron.set_fill(

                if abs(i - j) > 6:                    color=get_neuron_color(

                    continue                        neuron_fills[0][neuron_count],

                start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)                        vmax=np.abs(neuron_fills[0]).max(),

                line_grad = Line(start_point, end_point)                    ),

                line_grad.set_stroke(opacity=np.clip(grads_2_scaled[i, j], 0, 1), width=np.clip(1.0 * grads_2_scaled[i, j], 0, 3))                    opacity=1.0,

                line_grad.set_color(get_grad_color(grads_2_scaled[i, j]))                )

                grad_conections.add(line_grad)            neuron.move_to(

    return VGroup(connections, grad_conections, input_layer, hidden_layer, output_layer, dots)                LEFT * LAYER_SPACING

                + UP * ((INPUT_NEURONS // 2 - i) * VERTICAL_SPACING)

def get_attention_layer(attn_patterns):            )

    num_attention_pattern_slots = len(attn_patterns) + 1            input_layer.add(neuron)

    attention_pattern_spacing = 0.51            neuron_count += 1

    attention_border = RoundedRectangle(width=0.59, height=5.4, corner_radius=0.1)

    attention_border.set_stroke(width=1.0, color=FRESH_TAN)    neuron_count = 0

    attention_patterns = VGroup()    for i in range(HIDDEN_NEURONS):

    connection_points_left = VGroup()        if i == w1.shape[1] // 2:

    connection_points_right = VGroup()            dot = (

    attn_pattern_count = 0                Tex("...")

    for i in range(num_attention_pattern_slots):                .rotate(PI / 2, OUT)

        if i == num_attention_pattern_slots // 2:                .scale(DOTS_SCALE)

            dot = Tex("...").rotate(PI / 2, OUT).scale(0.5).move_to([0, num_attention_pattern_slots * attention_pattern_spacing / 2 - attention_pattern_spacing * (i + 0.5), 0])                .move_to(UP * ((HIDDEN_NEURONS // 2 - i) * VERTICAL_SPACING))

            dot.set_color(FRESH_TAN)            )

            attention_patterns.add(dot)            dot.set_color(neuron_stroke_color)

        else:            dots.add(dot)

            if i > num_attention_pattern_slots // 2:        else:

                offset = 0.15            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)

            else:            neuron.set_stroke(width=neuron_stroke_width)

                offset = -0.15            if neuron_fills is None:

            attn_pattern = AttentionPattern(matrix=attn_patterns[attn_pattern_count], square_size=0.07, stroke_width=0.5)                neuron.set_fill(color="#000000", opacity=1.0)

            attn_pattern.move_to([0, num_attention_pattern_slots * attention_pattern_spacing / 2 + offset - attention_pattern_spacing * (i + 0.5), 0])            else:

            attention_patterns.add(attn_pattern)                neuron.set_fill(

            connection_point_left = Circle(radius=0)                    color=get_neuron_color(

            connection_point_left.move_to([-0.59 / 2.0, num_attention_pattern_slots * attention_pattern_spacing / 2 + offset - attention_pattern_spacing * (i + 0.5), 0])                        neuron_fills[1][neuron_count],

            connection_points_left.add(connection_point_left)                        vmax=np.abs(neuron_fills[1]).max(),

            connection_point_right = Circle(radius=0)                    ),

            connection_point_right.move_to([0.59 / 2.0, num_attention_pattern_slots * attention_pattern_spacing / 2 + offset - attention_pattern_spacing * (i + 0.5), 0])                    opacity=1.0,

            connection_points_right.add(connection_point_right)                )

            attn_pattern_count += 1            neuron.move_to(UP * ((HIDDEN_NEURONS // 2 - i) * VERTICAL_SPACING))

    attention_layer = VGroup(attention_patterns, attention_border, connection_points_left, connection_points_right)            hidden_layer.add(neuron)

    return attention_layer            neuron_count += 1



def get_mlp_connections_left(attention_connections_left, mlp_out, connection_points_left, attention_connections_left_grad=None):    neuron_count = 0

    connections_left = VGroup()    for i in range(OUTPUT_NEURONS):

    attention_connections_left_abs = np.abs(attention_connections_left)        if i == w1.shape[0] // 2:

    attention_connections_left_scaled = attention_connections_left_abs / np.max(attention_connections_left_abs)            dot = (

    for i, mlp_out_neuron in enumerate(mlp_out):                Tex("...")

        for j, attention_neuron in enumerate(connection_points_left):                .rotate(PI / 2, OUT)

            if np.abs(attention_connections_left_scaled[i, j]) < 0.5:                .scale(DOTS_SCALE)

                continue                .move_to(

            if abs(i / 4 - j) > 3:                    RIGHT * LAYER_SPACING

                continue                    + UP * ((OUTPUT_NEURONS // 2 - i) * VERTICAL_SPACING)

            start_point, end_point = get_edge_points(mlp_out_neuron, attention_neuron, 0.06)                )

            line = Line(start_point, attention_neuron.get_center())            )

            line.set_stroke(opacity=np.clip(attention_connections_left_scaled[i, j], 0, 1), width=np.clip(1.0 * attention_connections_left_scaled[i, j], 0, 3))            dot.set_color(neuron_stroke_color)

            line.set_color(FRESH_TAN)            dots.add(dot)

            connections_left.add(line)        else:

    connections_left_grads = VGroup()            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)

    if attention_connections_left_grad is not None:            neuron.set_stroke(width=neuron_stroke_width)

        attention_connections_left_grad_abs = np.abs(attention_connections_left_grad)            if neuron_fills is None:

        attention_connections_left_grad_scaled = attention_connections_left_grad_abs / np.percentile(attention_connections_left_grad_abs, 98)                neuron.set_fill(color="#000000", opacity=1.0)

        for i, mlp_out_neuron in enumerate(mlp_out):            else:

            for j, attention_neuron in enumerate(connection_points_left):                neuron.set_fill(

                if np.abs(attention_connections_left_grad_scaled[i, j]) < 0.5:                    color=get_neuron_color(

                    continue                        neuron_fills[2][neuron_count],

                if abs(i / 4 - j) > 3:                        vmax=np.abs(neuron_fills[2]).max(),

                    continue                    ),

                start_point, end_point = get_edge_points(mlp_out_neuron, attention_neuron, 0.06)                    opacity=1.0,

                line = Line(start_point, attention_neuron.get_center())                )

                line.set_stroke(opacity=np.clip(attention_connections_left_grad_scaled[i, j], 0, 1), width=np.clip(1.0 * attention_connections_left_grad_scaled[i, j], 0, 2))            neuron.move_to(

                line.set_color(get_grad_color(attention_connections_left_grad_scaled[i, j]))                RIGHT * LAYER_SPACING

                connections_left_grads.add(line)                + UP * ((OUTPUT_NEURONS // 2 - i) * VERTICAL_SPACING)

    return connections_left, connections_left_grads            )

            output_layer.add(neuron)

def get_mlp_connections_right(attention_connections_right, mlp_in, connection_points_right, attention_connections_right_grad=None):            neuron_count += 1

    connections_right = VGroup()

    attention_connections_right_abs = np.abs(attention_connections_right)    connections = VGroup()

    attention_connections_right_scaled = attention_connections_right_abs / np.percentile(attention_connections_right_abs, 99)    w1_abs = np.abs(w1)

    for i, attention_neuron in enumerate(connection_points_right):    w1_scaled = w1_abs / np.percentile(w1_abs, 99)

        for j, mlp_in_neuron in enumerate(mlp_in):    for i, in_neuron in enumerate(input_layer):

            if np.abs(attention_connections_right_scaled[i, j]) < 0.6:        for j, hidden_neuron in enumerate(hidden_layer):

                continue            if np.abs(w1_scaled[i, j]) < 0.75:

            if abs(j / 4 - i) > 3:                continue

                continue            if abs(i - j) > 6:

            start_point, end_point = get_edge_points(mlp_in_neuron, attention_neuron, 0.06)                continue

            line = Line(start_point, attention_neuron.get_center())            start_point, end_point = get_edge_points(

            line.set_stroke(opacity=np.clip(attention_connections_right_scaled[i, j], 0, 1), width=np.clip(1.0 * attention_connections_right_scaled[i, j], 0, 3))                in_neuron, hidden_neuron, NEURON_RADIUS

            line.set_color(FRESH_TAN)            )

            connections_right.add(line)            line = Line(start_point, end_point)

    connections_right_grads = VGroup()

    if attention_connections_right_grad is not None:            line.set_stroke(

        attention_connections_right_grad_abs = np.abs(attention_connections_right_grad)                opacity=np.clip(w1_scaled[i, j], 0, 1), width=1.0 * w1_scaled[i, j]

        attention_connections_right_grad_scaled = attention_connections_right_grad_abs / np.percentile(attention_connections_right_grad_abs, 98)            )

        for i, attention_neuron in enumerate(connection_points_right):

            for j, mlp_in_neuron in enumerate(mlp_in):            line.set_color(line_stroke_color)

                if np.abs(attention_connections_right_grad_scaled[i, j]) < 0.5:            connections.add(line)

                    continue

                if abs(j / 4 - i) > 3:    w2_abs = np.abs(w2)

                    continue    w2_scaled = w2_abs / np.percentile(w2_abs, 99)

                start_point, end_point = get_edge_points(mlp_in_neuron, attention_neuron, 0.06)    for i, hidden_neuron in enumerate(hidden_layer):

                line = Line(start_point, attention_neuron.get_center())        for j, out_neuron in enumerate(output_layer):

                line.set_stroke(opacity=np.clip(attention_connections_right_grad_scaled[i, j], 0, 1), width=np.clip(1.0 * attention_connections_right_grad_scaled[i, j], 0, 3))            if np.abs(w2_scaled[i, j]) < 0.45:

                line.set_color(get_grad_color(attention_connections_right_grad_scaled[i, j]))                continue

                connections_right_grads.add(line)            if abs(i - j) > 6:

    return connections_right, connections_right_grads                continue

            start_point, end_point = get_edge_points(

def get_input_layer(prompt_neuron_indices, snapshot, num_input_neurons=36):                hidden_neuron, out_neuron, NEURON_RADIUS

    input_layer_neurons = VGroup()            )

    input_layer_text = VGroup()            line = Line(start_point, end_point)

    vertical_spacing = 0.18            line.set_stroke(

    neuron_radius = 0.06                opacity=np.clip(w2_scaled[i, j], 0, 1), width=1.0 * w2_scaled[i, j]

    neuron_stroke_color = "#dfd0b9"            )

    neuron_stroke_width = 1.0            line.set_color(line_stroke_color)

    words_to_nudge = {" capital": -0.02}            connections.add(line)

    prompt_token_count = 0

    neuron_count = 0    grad_conections = VGroup()

    for i in range(num_input_neurons):    if grads_1 is not None:

        if i == num_input_neurons // 2:        grads_1_abs = np.abs(grads_1)

            dot = Tex("...").rotate(PI / 2, OUT).scale(0.4).move_to(UP * ((num_input_neurons // 2 - i) * vertical_spacing))        grads_1_scaled = grads_1_abs / np.percentile(grads_1_abs, 95)

            dot.set_color(neuron_stroke_color)        for i, in_neuron in enumerate(input_layer):

        else:            for j, hidden_neuron in enumerate(hidden_layer):

            neuron = Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)                if np.abs(grads_1_scaled[i, j]) < 0.5:

            neuron.set_stroke(width=neuron_stroke_width)                    continue

            if neuron_count in prompt_neuron_indices:                if abs(i - j) > 6:

                neuron.set_fill(color="#dfd0b9", opacity=1.0)                    continue

                t = Text(snapshot["prompt.tokens"][prompt_token_count], font_size=24, font="myriad-pro")                start_point, end_point = get_edge_points(

                t.set_color(neuron_stroke_color)                    in_neuron, hidden_neuron, NEURON_RADIUS

                t.move_to((0.2 + t.get_right()[0]) * LEFT + UP * ((-t.get_bottom() + num_input_neurons // 2 - i) * vertical_spacing))                )

                token = snapshot["prompt.tokens"][prompt_token_count]                line_grad = Line(start_point, end_point)

                if token in words_to_nudge:                line_grad.set_stroke(

                    t.shift([0, words_to_nudge[token], 0])                    opacity=np.clip(grads_1_scaled[i, j], 0, 1),

                input_layer_text.add(t)                    width=np.clip(2.0 * grads_1_scaled[i, j], 0, 3),

                prompt_token_count += 1                )

            else:                line_grad.set_color(get_grad_color(grads_1_scaled[i, j]))

                neuron.set_fill(color="#000000", opacity=1.0)                grad_conections.add(line_grad)

            neuron.move_to(UP * ((num_input_neurons // 2 - i) * vertical_spacing))

            input_layer_neurons.add(neuron)    if grads_2 is not None:

            neuron_count += 1        grads_2_abs = np.abs(grads_2)

    input_layer = VGroup(input_layer_neurons, dot, input_layer_text)        grads_2_scaled = grads_2_abs / np.percentile(grads_2_abs, 97)

    return input_layer        for i, hidden_neuron in enumerate(hidden_layer):

            for j, out_neuron in enumerate(output_layer):

def get_output_layer(snapshot, empty=False):                if np.abs(grads_2_scaled[i, j]) < 0.5:

    output_layer_neurons = VGroup()                    continue

    output_layer_text = VGroup()                if abs(i - j) > 6:

    num_output_neurons = 36                    continue

    vertical_spacing = 0.18                start_point, end_point = get_edge_points(

    neuron_radius = 0.06                    hidden_neuron, out_neuron, NEURON_RADIUS

    neuron_stroke_color = "#dfd0b9"                )

    neuron_stroke_width = 1.0                line_grad = Line(start_point, end_point)

    neuron_count = 0                line_grad.set_stroke(

    for i in range(num_output_neurons):                    opacity=np.clip(grads_2_scaled[i, j], 0, 1),

        if i == num_output_neurons // 2:                    width=np.clip(1.0 * grads_2_scaled[i, j], 0, 3),

            dot = Tex("...").rotate(PI / 2, OUT).scale(0.4).move_to(UP * ((num_output_neurons // 2 - i) * vertical_spacing))                )

            dot.set_color(neuron_stroke_color)                line_grad.set_color(get_grad_color(grads_2_scaled[i, j]))

        else:                grad_conections.add(line_grad)

            n = Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)

            n.set_stroke(width=neuron_stroke_width)    return VGroup(

            if not empty:        connections, grad_conections, input_layer, hidden_layer, output_layer, dots

                n.set_fill(color=get_neuron_color(snapshot["topk.probs"][neuron_count], vmax=np.max(snapshot["topk.probs"])), opacity=1.0)    )

                if neuron_count == 0:

                    font_size = 22

                elif neuron_count < 4:def get_attention_layer(attn_patterns):

                    font_size = 16    num_attention_pattern_slots = len(attn_patterns) + 1

                else:    attention_pattern_spacing = 0.51

                    font_size = 12

                t = Text(snapshot["topk.tokens"][neuron_count], font_size=font_size, font="myriad-pro")    attention_border = RoundedRectangle(width=0.59, height=5.4, corner_radius=0.1)

                text_color = get_neuron_color(np.clip(snapshot["topk.probs"][neuron_count], 0.1, 1.0), vmax=np.max(snapshot["topk.probs"]))    attention_border.set_stroke(width=1.0, color=FRESH_TAN)

                t.set_color(text_color)

                t.set_opacity(np.clip(snapshot["topk.probs"][neuron_count], 0.3, 1.0))    attention_patterns = VGroup()

                t.move_to((0.2 + t.get_right()[0]) * RIGHT + UP * ((-t.get_bottom() + num_output_neurons // 2 - i) * vertical_spacing))    connection_points_left = VGroup()

                output_layer_text.add(t)    connection_points_right = VGroup()

            else:

                n.set_fill(color="#000000", opacity=1.0)    attn_pattern_count = 0

            n.move_to(UP * ((num_output_neurons // 2 - i) * vertical_spacing))    for i in range(num_attention_pattern_slots):

            output_layer_neurons.add(n)        if i == num_attention_pattern_slots // 2:

            neuron_count += 1            dot = (

    output_layer = VGroup(output_layer_neurons, dot, output_layer_text)                Tex("...")

    return output_layer                .rotate(PI / 2, OUT)

                .scale(0.5)

class P18(InteractiveScene):                .move_to(

    def construct(self):                    [

        layer_spacing = 0.22                        0,

        neuron_radius = 0.06                        num_attention_pattern_slots * attention_pattern_spacing / 2

        vertical_spacing = 0.14                        - attention_pattern_spacing * (i + 0.5),

        dots_scale = 0.25                        0,

        visible_neurons = 8                    ]

        neuron_layers = VGroup()                )

        dots = VGroup()            )

                    dot.set_color(FRESH_TAN)

        def get_vertical_positions():            attention_patterns.add(dot)

            half = visible_neurons // 2        else:

            top_y = [vertical_spacing * (half - i) for i in range(half)]            if i > num_attention_pattern_slots // 2:

            bottom_y = [vertical_spacing * (-(i + 1)) for i in range(half)]                offset = 0.15

            return top_y, bottom_y            else:

                        offset = -0.15

        for layer_idx in range(5):            attn_pattern = AttentionPattern(

            group = VGroup()                matrix=attn_patterns[attn_pattern_count],

            x_pos = layer_idx * layer_spacing                square_size=0.07,

            if layer_idx < 4:                stroke_width=0.5,

                top_y, bottom_y = get_vertical_positions()            )

                for y in top_y:            attn_pattern.move_to(

                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)                [

                    neuron.set_stroke(width=2)                    0,

                    neuron.set_fill(THUNDER, 1)                    num_attention_pattern_slots * attention_pattern_spacing / 2

                    neuron.set_stroke(WHITE, width=2)                    + offset

                    neuron.move_to(RIGHT * x_pos + UP * y)                    - attention_pattern_spacing * (i + 0.5),

                    group.add(neuron)                    0,

                dot = Tex("...").rotate(PI / 2, OUT).scale(dots_scale).set_color(FRESH_TAN).set_opacity(0.5)                ]

                dot.move_to(RIGHT * x_pos)            )

                dots.add(dot)            attention_patterns.add(attn_pattern)

                for y in bottom_y:

                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)            connection_point_left = Circle(radius=0)

                    neuron.set_stroke(width=2)            connection_point_left.move_to(

                    neuron.set_fill(THUNDER, 1)                [

                    neuron.set_stroke(WHITE, width=2)                    -0.59 / 2.0,

                    neuron.move_to(RIGHT * x_pos + UP * y)                    num_attention_pattern_slots * attention_pattern_spacing / 2

                    group.add(neuron)                    + offset

            else:                    - attention_pattern_spacing * (i + 0.5),

                top_y, bottom_y = get_vertical_positions()                    0,

                for y in top_y[2:]:                ]

                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)            )

                    neuron.set_stroke(width=2)            connection_points_left.add(connection_point_left)

                    neuron.set_fill(THUNDER, 1)

                    neuron.set_stroke(WHITE, width=2)            connection_point_right = Circle(radius=0)

                    neuron.move_to(RIGHT * x_pos + UP * y)            connection_point_right.move_to(

                    group.add(neuron)                [

                dot = Tex("...").rotate(PI / 2, OUT).scale(dots_scale).set_color(FRESH_TAN).set_opacity(0.5)                    0.59 / 2.0,

                dot.move_to(RIGHT * x_pos)                    num_attention_pattern_slots * attention_pattern_spacing / 2

                dots.add(dot)                    + offset

                for y in bottom_y[:-2]:                    - attention_pattern_spacing * (i + 0.5),

                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)                    0,

                    neuron.set_stroke(width=2)                ]

                    neuron.set_fill(THUNDER, 1)            )

                    neuron.set_stroke(WHITE, width=2)            connection_points_right.add(connection_point_right)

                    neuron.move_to(RIGHT * x_pos + UP * y)            attn_pattern_count += 1

                    group.add(neuron)

            neuron_layers.add(group)    attention_layer = VGroup(

                attention_patterns,

        all_neurons = VGroup()        attention_border,

        for layer in neuron_layers:        connection_points_left,

            all_neurons.add(*layer)        connection_points_right,

            )

        all_lines = VGroup()    return attention_layer

        neuron_to_lines = {}

        for neuron in all_neurons:

            neuron_to_lines[neuron] = VGroup()def get_mlp_connections_left(

            attention_connections_left,

        for layer_idx, layer in enumerate(neuron_layers):    mlp_out,

            if layer_idx < len(neuron_layers) - 1:    connection_points_left,

                lines = VGroup()    attention_connections_left_grad=None,

                current_layer = neuron_layers[layer_idx]):

                next_layer = neuron_layers[layer_idx + 1]    connections_left = VGroup()

                for neuron1 in current_layer:    attention_connections_left_abs = np.abs(attention_connections_left)

                    for neuron2 in next_layer:    attention_connections_left_scaled = attention_connections_left_abs / np.max(

                        line_segments = create_split_line(neuron1, neuron2, all_neurons, neuron_radius, exclude_neurons={neuron1, neuron2})        attention_connections_left_abs

                        for segment in line_segments:    )  # np.percentile(attention_connections_left_abs, 99)

                            segment.set_stroke(FRESH_TAN, width=2.0, opacity=0.6)    for i, mlp_out_neuron in enumerate(mlp_out):

                            neuron_to_lines[neuron1].add(segment)        for j, attention_neuron in enumerate(connection_points_left):

                            neuron_to_lines[neuron2].add(segment)            if np.abs(attention_connections_left_scaled[i, j]) < 0.5:

                        lines.add(line_segments)                continue

                all_lines.add(lines)            if abs(i / 4 - j) > 3:

                        continue  # Need to dial this up or lost it probably, but it is helpful!

        network = VGroup(all_lines, neuron_layers, dots)            start_point, end_point = get_edge_points(

        network.scale(4).move_to(ORIGIN)                mlp_out_neuron, attention_neuron, 0.06

        self.add(network)            )

        self.wait(2)            line = Line(start_point, attention_neuron.get_center())

                    # line.set_stroke(width=1, opacity=0.3)

        first_four_layers_neurons = VGroup()            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)

        for layer_idx in range(4):            line.set_stroke(

            first_four_layers_neurons.add(*neuron_layers[layer_idx])                opacity=np.clip(attention_connections_left_scaled[i, j], 0, 1),

                        width=np.clip(1.0 * attention_connections_left_scaled[i, j], 0, 3),

        for iteration in range(3):            )

            neurons_list = list(first_four_layers_neurons)            line.set_color(FRESH_TAN)

            total_neurons = len(neurons_list)            connections_left.add(line)

            num_to_disable = int(total_neurons * 0.3)

            indices_to_disable = np.random.choice(total_neurons, size=num_to_disable, replace=False)    connections_left_grads = VGroup()

            neurons_to_disable = [neurons_list[i] for i in indices_to_disable]    if attention_connections_left_grad is not None:

            disable_animations = []        attention_connections_left_grad_abs = np.abs(attention_connections_left_grad)

            for neuron in neurons_to_disable:        attention_connections_left_grad_scaled = (

                disable_animations.append(neuron.animate.set_opacity(0.2))            attention_connections_left_grad_abs

                for line_segment in neuron_to_lines[neuron]:            / np.percentile(attention_connections_left_grad_abs, 98)

                    disable_animations.append(line_segment.animate.set_opacity(0.2))        )

            self.play(*disable_animations, run_time=1.0)        for i, mlp_out_neuron in enumerate(mlp_out):

            self.wait(1)            for j, attention_neuron in enumerate(connection_points_left):

            enable_animations = []                if np.abs(attention_connections_left_grad_scaled[i, j]) < 0.5:

            for neuron in neurons_to_disable:                    continue

                enable_animations.append(neuron.animate.set_opacity(1.0))                if abs(i / 4 - j) > 3:

                for line_segment in neuron_to_lines[neuron]:                    continue

                    enable_animations.append(line_segment.animate.set_opacity(1.0))                start_point, end_point = get_edge_points(

            self.play(*enable_animations, run_time=1.0)                    mlp_out_neuron, attention_neuron, 0.06

            self.wait(1)                )

                        line = Line(start_point, attention_neuron.get_center())

        self.embed()                line.set_stroke(

                    opacity=np.clip(attention_connections_left_grad_scaled[i, j], 0, 1),

class P18_Long(InteractiveScene):                    width=np.clip(

    def construct(self):                        1.0 * attention_connections_left_grad_scaled[i, j], 0, 2

        layer_spacing = 0.22                    ),

        neuron_radius = 0.06                )

        vertical_spacing = 0.14                line.set_color(

        dots_scale = 0.25                    get_grad_color(attention_connections_left_grad_scaled[i, j])

        visible_neurons = 8                )

        neuron_layers = VGroup()                connections_left_grads.add(line)

        dots = VGroup()    return connections_left, connections_left_grads

        

        def get_vertical_positions():

            half = visible_neurons // 2def get_mlp_connections_right(

            top_y = [vertical_spacing * (half - i) for i in range(half)]    attention_connections_right,

            bottom_y = [vertical_spacing * (-(i + 1)) for i in range(half)]    mlp_in,

            return top_y, bottom_y    connection_points_right,

            attention_connections_right_grad=None,

        for layer_idx in range(5):):

            group = VGroup()    connections_right = VGroup()

            x_pos = layer_idx * layer_spacing    attention_connections_right_abs = np.abs(attention_connections_right)

            if layer_idx < 4:    attention_connections_right_scaled = (

                top_y, bottom_y = get_vertical_positions()        attention_connections_right_abs

                for y in top_y:        / np.percentile(attention_connections_right_abs, 99)

                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)    )

                    neuron.set_stroke(width=2)    for i, attention_neuron in enumerate(connection_points_right):

                    neuron.set_fill(THUNDER, 1)        for j, mlp_in_neuron in enumerate(mlp_in):

                    neuron.set_stroke(WHITE, width=2)            if np.abs(attention_connections_right_scaled[i, j]) < 0.6:

                    neuron.move_to(RIGHT * x_pos + UP * y)                continue

                    group.add(neuron)            if abs(j / 4 - i) > 3:

                dot = Tex("...").rotate(PI / 2, OUT).scale(dots_scale).set_color(FRESH_TAN).set_opacity(0.5)                continue  # Need to dial this up or lost it probably, but it is helpful!

                dot.move_to(RIGHT * x_pos)            start_point, end_point = get_edge_points(

                dots.add(dot)                mlp_in_neuron, attention_neuron, 0.06

                for y in bottom_y:            )

                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)            line = Line(start_point, attention_neuron.get_center())

                    neuron.set_stroke(width=2)            # line.set_stroke(width=1, opacity=0.3)

                    neuron.set_fill(THUNDER, 1)            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)

                    neuron.set_stroke(WHITE, width=2)            line.set_stroke(

                    neuron.move_to(RIGHT * x_pos + UP * y)                opacity=np.clip(attention_connections_right_scaled[i, j], 0, 1),

                    group.add(neuron)                width=np.clip(1.0 * attention_connections_right_scaled[i, j], 0, 3),

            else:            )

                top_y, bottom_y = get_vertical_positions()            line.set_color(FRESH_TAN)

                for y in top_y[2:]:            connections_right.add(line)

                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)

                    neuron.set_stroke(width=2)    connections_right_grads = VGroup()

                    neuron.set_fill(THUNDER, 1)    if attention_connections_right_grad is not None:

                    neuron.set_stroke(WHITE, width=2)        attention_connections_right_grad_abs = np.abs(attention_connections_right_grad)

                    neuron.move_to(RIGHT * x_pos + UP * y)        attention_connections_right_grad_scaled = (

                    group.add(neuron)            attention_connections_right_grad_abs

                dot = Tex("...").rotate(PI / 2, OUT).scale(dots_scale).set_color(FRESH_TAN).set_opacity(0.5)            / np.percentile(attention_connections_right_grad_abs, 98)

                dot.move_to(RIGHT * x_pos)        )

                dots.add(dot)        for i, attention_neuron in enumerate(connection_points_right):

                for y in bottom_y[:-2]:            for j, mlp_in_neuron in enumerate(mlp_in):

                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)                if np.abs(attention_connections_right_grad_scaled[i, j]) < 0.5:

                    neuron.set_stroke(width=2)                    continue

                    neuron.set_fill(THUNDER, 1)                if abs(j / 4 - i) > 3:

                    neuron.set_stroke(WHITE, width=2)                    continue

                    neuron.move_to(RIGHT * x_pos + UP * y)                start_point, end_point = get_edge_points(

                    group.add(neuron)                    mlp_in_neuron, attention_neuron, 0.06

            neuron_layers.add(group)                )

                        line = Line(start_point, attention_neuron.get_center())

        all_neurons = VGroup()                line.set_stroke(

        for layer in neuron_layers:                    opacity=np.clip(attention_connections_right_grad_scaled[i, j], 0, 1),

            all_neurons.add(*layer)                    width=np.clip(

                                1.0 * attention_connections_right_grad_scaled[i, j], 0, 3

        all_lines = VGroup()                    ),

        neuron_to_lines = {}                )

        for neuron in all_neurons:                line.set_color(

            neuron_to_lines[neuron] = VGroup()                    get_grad_color(attention_connections_right_grad_scaled[i, j])

                        )

        for layer_idx, layer in enumerate(neuron_layers):                connections_right_grads.add(line)

            if layer_idx < len(neuron_layers) - 1:    return connections_right, connections_right_grads

                lines = VGroup()

                current_layer = neuron_layers[layer_idx]

                next_layer = neuron_layers[layer_idx + 1]def get_input_layer(prompt_neuron_indices, snapshot, num_input_neurons=36):

                for neuron1 in current_layer:    input_layer_neurons = VGroup()

                    for neuron2 in next_layer:    input_layer_text = VGroup()

                        line_segments = create_split_line(neuron1, neuron2, all_neurons, neuron_radius, exclude_neurons={neuron1, neuron2})    vertical_spacing = 0.18

                        for segment in line_segments:    neuron_radius = 0.06

                            segment.set_stroke(FRESH_TAN, width=2.0, opacity=0.6)    neuron_stroke_color = "#dfd0b9"

                            neuron_to_lines[neuron1].add(segment)    neuron_stroke_width = 1.0

                            neuron_to_lines[neuron2].add(segment)    words_to_nudge = {" capital": -0.02}

                        lines.add(line_segments)

                all_lines.add(lines)    prompt_token_count = 0

            neuron_count = 0

        network = VGroup(all_lines, neuron_layers, dots)    for i in range(num_input_neurons):

        network.scale(4).move_to(ORIGIN)        if i == num_input_neurons // 2:

        self.add(network)            dot = (

        self.wait(2)                Tex("...")

                        .rotate(PI / 2, OUT)

        first_four_layers_neurons = VGroup()                .scale(0.4)

        for layer_idx in range(4):                .move_to(UP * ((num_input_neurons // 2 - i) * vertical_spacing))

            first_four_layers_neurons.add(*neuron_layers[layer_idx])            )

                    dot.set_color(neuron_stroke_color)

        for iteration in range(20):        else:

            neurons_list = list(first_four_layers_neurons)            neuron = Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)

            total_neurons = len(neurons_list)            neuron.set_stroke(width=neuron_stroke_width)

            num_to_disable = int(total_neurons * 0.3)            if neuron_count in prompt_neuron_indices:

            indices_to_disable = np.random.choice(total_neurons, size=num_to_disable, replace=False)                neuron.set_fill(color="#dfd0b9", opacity=1.0)

            neurons_to_disable = [neurons_list[i] for i in indices_to_disable]                t = Text(

            disable_animations = []                    snapshot["prompt.tokens"][prompt_token_count],

            for neuron in neurons_to_disable:                    font_size=24,

                disable_animations.append(neuron.animate.set_opacity(0.2))                    font="myriad-pro",

                for line_segment in neuron_to_lines[neuron]:                )

                    disable_animations.append(line_segment.animate.set_opacity(0.2))                t.set_color(neuron_stroke_color)

            self.play(*disable_animations, run_time=1.0)                t.move_to(

            self.wait(1)                    (0.2 + t.get_right()[0]) * LEFT

            enable_animations = []                    + UP

            for neuron in neurons_to_disable:                    * (

                enable_animations.append(neuron.animate.set_opacity(1.0))                        (-t.get_bottom() + num_input_neurons // 2 - i)

                for line_segment in neuron_to_lines[neuron]:                        * vertical_spacing

                    enable_animations.append(line_segment.animate.set_opacity(1.0))                    )

            self.play(*enable_animations, run_time=1.0)                )

            self.wait(1)                token = snapshot["prompt.tokens"][prompt_token_count]

                        if token in words_to_nudge:

        self.embed()                    t.shift([0, words_to_nudge[token], 0])


                input_layer_text.add(t)
                prompt_token_count += 1
            else:
                neuron.set_fill(color="#000000", opacity=1.0)

            neuron.move_to(UP * ((num_input_neurons // 2 - i) * vertical_spacing))
            input_layer_neurons.add(neuron)
            neuron_count += 1

    input_layer = VGroup(input_layer_neurons, dot, input_layer_text)
    return input_layer


def get_output_layer(snapshot, empty=False):
    output_layer_neurons = VGroup()
    output_layer_text = VGroup()
    num_output_neurons = 36
    vertical_spacing = 0.18
    neuron_radius = 0.06
    neuron_stroke_color = "#dfd0b9"
    neuron_stroke_width = 1.0

    neuron_count = 0
    for i in range(num_output_neurons):
        if i == num_output_neurons // 2:
            dot = (
                Tex("...")
                .rotate(PI / 2, OUT)
                .scale(0.4)
                .move_to(UP * ((num_output_neurons // 2 - i) * vertical_spacing))
            )
            dot.set_color(neuron_stroke_color)
        else:
            n = Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)
            n.set_stroke(width=neuron_stroke_width)
            if not empty:
                n.set_fill(
                    color=get_neuron_color(
                        snapshot["topk.probs"][neuron_count],
                        vmax=np.max(snapshot["topk.probs"]),
                    ),
                    opacity=1.0,
                )
                if neuron_count == 0:
                    font_size = 22
                elif neuron_count < 4:
                    font_size = 16
                else:
                    font_size = 12
                t = Text(
                    snapshot["topk.tokens"][neuron_count],
                    font_size=font_size,
                    font="myriad-pro",
                )
                text_color = get_neuron_color(
                    np.clip(snapshot["topk.probs"][neuron_count], 0.1, 1.0),
                    vmax=np.max(snapshot["topk.probs"]),
                )
                t.set_color(text_color)
                t.set_opacity(np.clip(snapshot["topk.probs"][neuron_count], 0.3, 1.0))
                t.move_to(
                    (0.2 + t.get_right()[0]) * RIGHT
                    + UP
                    * (
                        (-t.get_bottom() + num_output_neurons // 2 - i)
                        * vertical_spacing
                    )
                )
                output_layer_text.add(t)

            else:
                n.set_fill(color="#000000", opacity=1.0)

            n.move_to(UP * ((num_output_neurons // 2 - i) * vertical_spacing))
            output_layer_neurons.add(n)
            neuron_count += 1
    output_layer = VGroup(output_layer_neurons, dot, output_layer_text)
    return output_layer

class P18(InteractiveScene):
    def construct(self):
        layer_spacing = 0.22
        neuron_radius = 0.06
        vertical_spacing = 0.14
        dots_scale = 0.25
        visible_neurons = 8

        neuron_layers = VGroup()
        dots = VGroup()

        def get_vertical_positions():
            half = visible_neurons // 2
            top_y = [vertical_spacing * (half - i) for i in range(half)]
            bottom_y = [vertical_spacing * (-(i + 1)) for i in range(half)]
            return top_y, bottom_y

        for layer_idx in range(5):
            group = VGroup()
            x_pos = layer_idx * layer_spacing
            if layer_idx < 4:
                top_y, bottom_y = get_vertical_positions()
                for y in top_y:
                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(WHITE, width=2)
                    neuron.move_to(RIGHT * x_pos + UP * y)
                    group.add(neuron)
                dot = (
                    Tex("...")
                    .rotate(PI / 2, OUT)
                    .scale(dots_scale)
                    .set_color(FRESH_TAN)
                ).set_opacity(0.5)
                dot.move_to(RIGHT * x_pos)
                dots.add(dot)
                for y in bottom_y:
                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(WHITE, width=2)
                    neuron.move_to(RIGHT * x_pos + UP * y)
                    group.add(neuron)
            else:
                top_y, bottom_y = get_vertical_positions()
                for y in top_y[2:]:
                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(WHITE, width=2)
                    neuron.move_to(RIGHT * x_pos + UP * y)
                    group.add(neuron)
                dot = (
                    Tex("...")
                    .rotate(PI / 2, OUT)
                    .scale(dots_scale)
                    .set_color(FRESH_TAN)
                ).set_opacity(0.5)
                dot.move_to(RIGHT * x_pos)
                dots.add(dot)
                for y in bottom_y[:-2]:
                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(WHITE, width=2)
                    neuron.move_to(RIGHT * x_pos + UP * y)
                    group.add(neuron)
            neuron_layers.add(group)

        all_neurons = VGroup()
        for layer in neuron_layers:
            all_neurons.add(*layer)

        all_lines = VGroup()
        neuron_to_lines = {}
        
        for neuron in all_neurons:
            neuron_to_lines[neuron] = VGroup()

        for layer_idx, layer in enumerate(neuron_layers):
            if layer_idx < len(neuron_layers) - 1:
                lines = VGroup()
                current_layer = neuron_layers[layer_idx]
                next_layer = neuron_layers[layer_idx + 1]

                for neuron1 in current_layer:
                    for neuron2 in next_layer:
                        line_segments = create_split_line(
                            neuron1,
                            neuron2,
                            all_neurons,
                            neuron_radius,
                            exclude_neurons={neuron1, neuron2}
                        )

                        for segment in line_segments:
                            segment.set_stroke(FRESH_TAN, width=2.0, opacity=0.6)
                            neuron_to_lines[neuron1].add(segment)
                            neuron_to_lines[neuron2].add(segment)

                        lines.add(line_segments)

                all_lines.add(lines)
                
        network = VGroup(all_lines, neuron_layers, dots)
        network.scale(4).move_to(ORIGIN)

        self.add(network)
        self.wait(2)

        first_four_layers_neurons = VGroup()
        for layer_idx in range(4):
            first_four_layers_neurons.add(*neuron_layers[layer_idx])

        for iteration in range(3):
            neurons_list = list(first_four_layers_neurons)
            total_neurons = len(neurons_list)
            num_to_disable = int(total_neurons * 0.3)
            indices_to_disable = np.random.choice(
                total_neurons, 
                size=num_to_disable, 
                replace=False
            )
            neurons_to_disable = [neurons_list[i] for i in indices_to_disable]

            disable_animations = []
            for neuron in neurons_to_disable:
                disable_animations.append(neuron.animate.set_opacity(0.2))
                for line_segment in neuron_to_lines[neuron]:
                    disable_animations.append(line_segment.animate.set_opacity(0.2))

            self.play(*disable_animations, run_time=1.0)
            self.wait(1)

            enable_animations = []
            for neuron in neurons_to_disable:
                enable_animations.append(neuron.animate.set_opacity(1.0))
                for line_segment in neuron_to_lines[neuron]:
                    enable_animations.append(line_segment.animate.set_opacity(1.0))

            self.play(*enable_animations, run_time=1.0)
            self.wait(1)

        self.embed()

class P18_Long(InteractiveScene):
    def construct(self):
        layer_spacing = 0.22
        neuron_radius = 0.06
        vertical_spacing = 0.14
        dots_scale = 0.25
        visible_neurons = 8

        neuron_layers = VGroup()
        dots = VGroup()

        def get_vertical_positions():
            half = visible_neurons // 2
            top_y = [vertical_spacing * (half - i) for i in range(half)]
            bottom_y = [vertical_spacing * (-(i + 1)) for i in range(half)]
            return top_y, bottom_y

        for layer_idx in range(5):
            group = VGroup()
            x_pos = layer_idx * layer_spacing
            if layer_idx < 4:
                top_y, bottom_y = get_vertical_positions()
                for y in top_y:
                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(WHITE, width=2)
                    neuron.move_to(RIGHT * x_pos + UP * y)
                    group.add(neuron)
                dot = (
                    Tex("...")
                    .rotate(PI / 2, OUT)
                    .scale(dots_scale)
                    .set_color(FRESH_TAN)
                ).set_opacity(0.5)
                dot.move_to(RIGHT * x_pos)
                dots.add(dot)
                for y in bottom_y:
                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(WHITE, width=2)
                    neuron.move_to(RIGHT * x_pos + UP * y)
                    group.add(neuron)
            else:
                top_y, bottom_y = get_vertical_positions()
                for y in top_y[2:]:
                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(WHITE, width=2)
                    neuron.move_to(RIGHT * x_pos + UP * y)
                    group.add(neuron)
                dot = (
                    Tex("...")
                    .rotate(PI / 2, OUT)
                    .scale(dots_scale)
                    .set_color(FRESH_TAN)
                ).set_opacity(0.5)
                dot.move_to(RIGHT * x_pos)
                dots.add(dot)
                for y in bottom_y[:-2]:
                    neuron = Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(WHITE, width=2)
                    neuron.move_to(RIGHT * x_pos + UP * y)
                    group.add(neuron)
            neuron_layers.add(group)

        all_neurons = VGroup()
        for layer in neuron_layers:
            all_neurons.add(*layer)

        all_lines = VGroup()
        neuron_to_lines = {}
        
        for neuron in all_neurons:
            neuron_to_lines[neuron] = VGroup()

        for layer_idx, layer in enumerate(neuron_layers):
            if layer_idx < len(neuron_layers) - 1:
                lines = VGroup()
                current_layer = neuron_layers[layer_idx]
                next_layer = neuron_layers[layer_idx + 1]

                for neuron1 in current_layer:
                    for neuron2 in next_layer:
                        line_segments = create_split_line(
                            neuron1,
                            neuron2,
                            all_neurons,
                            neuron_radius,
                            exclude_neurons={neuron1, neuron2}
                        )

                        for segment in line_segments:
                            segment.set_stroke(FRESH_TAN, width=2.0, opacity=0.6)
                            neuron_to_lines[neuron1].add(segment)
                            neuron_to_lines[neuron2].add(segment)

                        lines.add(line_segments)

                all_lines.add(lines)
                
        network = VGroup(all_lines, neuron_layers, dots)
        network.scale(4).move_to(ORIGIN)

        self.add(network)
        self.wait(2)

        first_four_layers_neurons = VGroup()
        for layer_idx in range(4):
            first_four_layers_neurons.add(*neuron_layers[layer_idx])

        for iteration in range(20):
            neurons_list = list(first_four_layers_neurons)
            total_neurons = len(neurons_list)
            num_to_disable = int(total_neurons * 0.3)
            indices_to_disable = np.random.choice(
                total_neurons, 
                size=num_to_disable, 
                replace=False
            )
            neurons_to_disable = [neurons_list[i] for i in indices_to_disable]

            disable_animations = []
            for neuron in neurons_to_disable:
                disable_animations.append(neuron.animate.set_opacity(0.2))
                for line_segment in neuron_to_lines[neuron]:
                    disable_animations.append(line_segment.animate.set_opacity(0.2))

            self.play(*disable_animations, run_time=1.0)
            self.wait(1)

            enable_animations = []
            for neuron in neurons_to_disable:
                enable_animations.append(neuron.animate.set_opacity(1.0))
                for line_segment in neuron_to_lines[neuron]:
                    enable_animations.append(line_segment.animate.set_opacity(1.0))

            self.play(*enable_animations, run_time=1.0)
            self.wait(1)

        self.embed()
