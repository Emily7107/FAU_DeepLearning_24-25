from pattern import Checker
from pattern import Circle
from pattern import Spectrum
from generator import ImageGenerator


def generate_checkerboard():
    checker_size = 100
    num_squares = 10
    checker = Checker(checker_size, num_squares)
    checker.draw()
    checker.show()


def generate_circle():
    circle_size = 1024
    radius = 200
    position = (512, 256)
    circle = Circle(circle_size, radius, position)
    circle.draw()
    circle.show()

def generate_spectrum():
    s = Spectrum(255)
    s.draw()
    s.show()
    
def generate_image():
    file_path = './exercise_data/'
    label_path = './Labels.json'
    batch_size = 12
    image_size = [32, 32, 3]  # height, weight, channel_num

    img_gen = ImageGenerator(file_path, label_path, batch_size,
                             image_size, rotation=True, mirroring=True, shuffle=True)

    img_gen.show()


def main():
    generate_checkerboard()
    generate_circle()
    generate_image()


if __name__ == "__main__":
    main()