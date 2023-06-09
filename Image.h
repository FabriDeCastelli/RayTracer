/**
 * @file Image.h
 */

//
//  Image.h
//  Raytracer
//

#ifndef Image_h
#define Image_h

using namespace std;

/**
 * @class Image.
 * @brief Represents an image.
 *
 * Allows to create an image and write it to a file.
 */
class Image
{

private:
    /**
     * Width of the image.
     */
    int width;

    /**
     * Height of the image.
     */
    int height;

    /**
     * Pointer to the data representing the images.
     */
    int *data;

public:
    /**
     * Constructor.
     *
     * @param width width of the image
     * @param height height of the image
     */
    Image(int width, int height) : width(width), height(height)
    {
        data = new int[3 * width * height];
    }

    /**
     * Writes and image to a file in ppm format.
     *
     * @param path the path where to the target image
     */
    void writeImage(const char *path)
    {
        ofstream file;
        file.open(path);
        file << "P3" << endl;
        file << width << " " << height << endl;
        file << 255 << endl;

        for (int h = 0; h < height; h++)
        {

            for (int w = 0; w < width; w++)
            {
                file << data[3 * (h * width + w)] << " ";
                file << data[3 * (h * width + w) + 1] << " ";
                file << data[3 * (h * width + w) + 2] << "  ";
            }

            file << endl;
        }

        file.close();
    }

    /**
     * Set a value for one pixel.
     *
     * @param x x coordinate of the pixel - index of the column counting from left to right
     * @param y y coordinate of the pixel - index of the row counting from top to bottom
     * @param r red chanel value in range from 0 to 255
     * @param g green chanel value in range from 0 to 255
     * @param b blue chanel value in range from 0 to 255
     */
    void setPixel(int x, int y, int r, int g, int b)
    {
        data[3 * (y * width + x)] = r;
        data[3 * (y * width + x) + 1] = g;
        data[3 * (y * width + x) + 2] = b;
    }

    /**
     * Set a value for one pixel.
     *
     * @param x x coordinate of the pixel - index of the column counting from left to right
     * @param y y coordinate of the pixel - index of the row counting from top to bottom
     * @param r red chanel value in range from 0 to 1
     * @param g green chanel value in range from 0 to 1
     * @param b blue chanel value in range from 0 to 1
     */
    void setPixel(int x, int y, float r, float g, float b)
    {
        data[3 * (y * width + x)] = (float)(255 * r);
        data[3 * (y * width + x) + 1] = (float)(255 * g);
        data[3 * (y * width + x) + 2] = (float)(255 * b);
    }

    /**
     * Set a value for one pixel.
     *
     * @param x x coordinate of the pixel - index of the column counting from left to right
     * @param y y coordinate of the pixel - index of the row counting from top to bottom
     * @param color color of the pixel expressed as vec3 of RGB values in range from 0 to 1
     */
    void setPixel(int x, int y, glm::vec3 color)
    {
        data[3 * (y * width + x)] = (float)(255 * color.r);
        data[3 * (y * width + x) + 1] = (float)(255 * color.g);
        data[3 * (y * width + x) + 2] = (float)(255 * color.b);
    }
};

#endif /* Image_h */
