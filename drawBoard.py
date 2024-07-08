import tkinter as tk

def draw_triangle(canvas, x, y, size, color):
    # Different Triangle Vertices
    points = [
        x, y + size,  # Bottom-left
        x + size, y + size,  # Bottom-right
        x + size / 2, y  # Top
    ]
    canvas.create_polygon(points, fill=color, outline='black')

'''
def draw_octagon(canvas, x, y, size, color):
    # Calculate the vertices of the octagon
    r = size / (2 * (1 + (2**0.5)))
    points = [
        x + r, y,  # Top
        x + size - r, y,  # Top-right
        x + size, y + r,  # Right-top
        x + size, y + size - r,  # Right-bottom
        x + size - r, y + size,  # Bottom-right
        x + r, y + size,  # Bottom-left
        x, y + size - r,  # Left-bottom
        x, y + r  # Left-top
    ]
    canvas.create_polygon(points, fill=color, outline='black')
'''

def draw_checkerboard(canvas, n, tileSize):
    for i in range(n):
        for j in range(n):
            x, y = i * tileSize, j * tileSize
            if (i + j) % 2 == 0:
                canvas.create_rectangle(x, y, x + tileSize, y + tileSize, fill='white', outline='black')
                #draw_triangle(canvas, x, y, tileSize, 'black')
            else:
                canvas.create_rectangle(x, y, x + tileSize, y + tileSize, fill='black', outline='black')
                #draw_octagon(canvas, x, y, tileSize, 'white')

root = tk.Tk()
root.title("Checkerboard")

canvas_size = 400
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size)
canvas.pack()

n = 8
tileSize = canvas_size // n
draw_checkerboard(canvas, n, tileSize)

root.mainloop()
