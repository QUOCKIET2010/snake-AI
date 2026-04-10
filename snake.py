class Snake:
    def __init__(self, start_pos=(5, 5)):
        self.body = [start_pos, (start_pos[0]-1, start_pos[1]), (start_pos[0]-2, start_pos[1])]
        self.direction = (1, 0)
        self.alive = True

    def head(self):
        return self.body[0]

    def move(self, new_head):
        self.body.insert(0, new_head)
        
    def shrink(self):
        self.body.pop()