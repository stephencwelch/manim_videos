from manimlib import *
import json

class Node(VGroup):
    def __init__(self, node_data, radius=0.3, font_size=24, **kwargs):
        super().__init__(**kwargs)

        self.node_id = node_data["id"]
        self.phoneme = node_data["phoneme"]
        self.connects_from = node_data["connects_from"]

        self.circle = Circle(radius=radius, color=WHITE)
        self.text = Text(self.phoneme, font_size=font_size)

        self.add(self.circle, self.text)


class Connection(VGroup):
    def __init__(self, node_from, node_to, **kwargs):
        super().__init__(**kwargs)

        self.node_from = node_from
        self.node_to = node_to

        self.arrow = Arrow(
            node_from.get_center(),
            node_to.get_center(),
            buff=0.35,
            color=WHITE
        )

        self.add(self.arrow)


class TestNode(Scene):
    def construct(self):
        node_data = {
            "id": 0,
            "phoneme": "T",
            "connects_from": []
        }
        node = Node(node_data)
        self.add(node)
        self.wait()


class TestConnection(Scene):
    def construct(self):
        node1_data = {
            "id": 0,
            "phoneme": "T",
            "connects_from": []
        }
        node2_data = {
            "id": 1,
            "phoneme": "E",
            "connects_from": [0]
        }

        node1 = Node(node1_data).shift(LEFT * 2)
        node2 = Node(node2_data).shift(RIGHT * 2)
        connection = Connection(node1, node2)

        self.add(node1, node2, connection)
        self.wait()
