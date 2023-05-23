/**
 * @file main.cpp
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

#include "Image.h"
#include "Material.h"

using namespace std;

/**
 * @class Ray.
 * @brief Represents a single Ray.
 */
class Ray
{

public:
	/**
	 * Origin of the ray.
	 */
	glm::vec3 origin;

	/**
	 * Direction of the ray.
	 */
	glm::vec3 direction;

	/**
	 * Constructor of the ray.
	 *
	 * @param origin origin of the ray
	 * @param direction direction of the ray
	 */
	Ray(glm::vec3 origin, glm::vec3 direction) : origin(origin), direction(direction) {}
};

class Object;

/**
 * @struct Hit
 * @brief Represents the intersection of a ray with an object.
 *
 * Contains all information about the intersection of a ray with an object, if it exists.
 */
struct Hit
{

	/**
	 * Whether there was an intersection.
	 */
	bool hit;

	/**
	 * Normal vector of the intersected object at the intersection point.
	 */
	glm::vec3 normal;

	/**
	 * Point of intersection.
	 */
	glm::vec3 intersection;

	/**
	 * Distance from the origin of the ray to the intersection point.
	 */
	float distance;

	/**
	 * Object that was intersected.
	 */
	Object *object;

	/**
	 * Texture coordinates of the intersection point.
	 */
	glm::vec2 uv;
};

/**
 * @class Object.
 * @brief Represents an object in the scene.
 */
class Object
{

protected:
	/**
	 * Transformation matrix from the local to the global coordinate system.
	 */
	glm::mat4 transformationMatrix;

	/**
	 * Inverse of the transformation matrix from global to local coordinate system.
	 */
	glm::mat4 inverseTransformationMatrix;

	/**
	 * Normal matrix for transforming normal vectors from the local to the global coordinate system.
	 */
	glm::mat4 normalMatrix;

public:
	/**
	 * Color of the object.
	 */
	glm::vec3 color;

	/**
	 * Material of the object.
	 */
	Material material;

	/**
	 * Computes an intersection.
	 *
	 * @param ray A ray that intersects the object
	 * @return The hit information
	 */
	virtual Hit intersect(Ray ray) = 0;

	/**
	 * Getter for material.
	 *
	 * @return The material of this object.
	 */
	Material getMaterial()
	{
		return material;
	}

	/**
	 * Setter for the material.
	 *
	 * @param material the new material
	 */
	void setMaterial(Material material)
	{
		this->material = material;
	}

	/**
	 * Setter for the transformation matrices.
	 *
	 * @param matrix the new transformation matrix
	 */
	void setTransformation(glm::mat4 matrix)
	{
		transformationMatrix = matrix;
		inverseTransformationMatrix = glm::inverse(matrix);
		normalMatrix = glm::transpose(inverseTransformationMatrix);
	}
};

/**
 * @class Sphere.
 * @brief Represents a sphere in the scene.
 *
 * A sphere is defined by its radius and its center.
 */
class Sphere : public Object
{

private:
	/**
	 * Radius of the sphere.
	 */
	float radius;

	/**
	 * Center of the sphere.
	 */
	glm::vec3 center;

public:
	/**
	 * Constructor of the sphere.
	 *
	 * @param radius radius of the sphere
	 * @param center center of the sphere
	 * @param color color of the sphere
	 *
	 */
	Sphere(float radius, glm::vec3 center, glm::vec3 color) : radius(radius), center(center)
	{
		this->color = color;
	}

	/**
	 * Constructor of the sphere.
	 *
	 * @param radius radius of the sphere
	 * @param center center of the sphere
	 * @param material material of the sphere
	 */
	Sphere(float radius, glm::vec3 center, Material material) : radius(radius), center(center)
	{
		this->material = material;
	}
	/**
	 * Computes an intersection.
	 *
	 * @param ray A ray that intersects the sphere
	 *
	 * @return The hit information
	 */
	Hit intersect(Ray ray)
	{

		glm::vec3 c = center - ray.origin;

		float cdotc = glm::dot(c, c);
		float cdotd = glm::dot(c, ray.direction);

		Hit hit;

		float D = 0;
		if (cdotc > cdotd * cdotd)
		{
			D = sqrt(cdotc - cdotd * cdotd);
		}

		if (D <= radius)
		{
			hit.hit = true;
			float t1 = cdotd - sqrt(radius * radius - D * D);
			float t2 = cdotd + sqrt(radius * radius - D * D);

			float t = t1;
			if (t < 0)
				t = t2;
			if (t < 0)
			{
				hit.hit = false;
				return hit;
			}

			hit.intersection = ray.origin + t * ray.direction;
			hit.normal = glm::normalize(hit.intersection - center);
			hit.distance = glm::distance(ray.origin, hit.intersection);
			hit.object = this;

			hit.uv.s = (asin(hit.normal.y) + M_PI / 2) / M_PI;
			hit.uv.t = (atan2(hit.normal.z, hit.normal.x) + M_PI) / (2 * M_PI);
		}
		else
		{
			hit.hit = false;
		}
		return hit;
	}
};

/**
 * @class Plane.
 * @brief Represents a plane in the scene.
 *
 * A plane is defined by a point and a normal vector.
 */
class Plane : public Object
{

private:
	/**
	 * Normal vector of the plane.
	 */
	glm::vec3 normal;

	/**
	 * Point on the plane.
	 */
	glm::vec3 point;

public:
	/**
	 * Constructor of the plane.
	 *
	 * @param point point on the plane
	 * @param normal normal vector of the plane
	 */
	Plane(glm::vec3 point, glm::vec3 normal) : point(point), normal(normal) {}

	/**
	 * Constructor of the plane.
	 *
	 * @param point point on the plane
	 * @param normal normal vector of the plane
	 * @param material material of the plane
	 */
	Plane(glm::vec3 point, glm::vec3 normal, Material material) : point(point), normal(normal)
	{
		this->material = material;
	}

	/**
	 * Computes an intersection.
	 *
	 * @param ray A ray that intersects the plane
	 *
	 * @return The hit information
	 */
	Hit intersect(Ray ray)
	{

		Hit hit;
		hit.hit = false;
		float DdotN = glm::dot(ray.direction, normal);
		if (DdotN < 0)
		{

			float PdotN = glm::dot(point - ray.origin, normal);
			float t = PdotN / DdotN;

			if (t > 0)
			{
				hit.hit = true;
				hit.normal = normal;
				hit.distance = t;
				hit.object = this;
				hit.intersection = t * ray.direction + ray.origin;
			}
		}
		return hit;
	}
};

/**
 * @class Cone.
 * @brief Represents a cone in the scene.
 *
 * A cone is represented by the plane (0, 1, 0), its vertex is at (0, 0, 0) and its height is 1.
 */
class Cone : public Object
{
private:
	/**
	 * The plane that represents the cone.
	 */
	Plane *plane;

public:
	/**
	 * Constructor of the cone.
	 *
	 * @param material material of the cone
	 */
	Cone(Material material)
	{
		this->material = material;
		plane = new Plane(glm::vec3(0, 1, 0), glm::vec3(0, 1, 0));
	}

	/**
	 * Computes an intersection.
	 *
	 * @param ray A ray that intersects the cone
	 *
	 * @return The hit information
	 */
	Hit intersect(Ray ray)
	{

		Hit hit;
		hit.hit = false;

		glm::vec3 d = inverseTransformationMatrix * glm::vec4(ray.direction, 0.0); // implicit cast to vec3
		glm::vec3 o = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0);	   // implicit cast to vec3
		d = glm::normalize(d);

		float a = d.x * d.x + d.z * d.z - d.y * d.y;
		float b = 2 * (d.x * o.x + d.z * o.z - d.y * o.y);
		float c = o.x * o.x + o.z * o.z - o.y * o.y;

		float delta = b * b - 4 * a * c;

		if (delta < 0)
		{
			return hit;
		}

		float t1 = (-b - sqrt(delta)) / (2 * a);
		float t2 = (-b + sqrt(delta)) / (2 * a);

		float t = t1;
		hit.intersection = o + t * d;
		if (t < 0 || hit.intersection.y > 1 || hit.intersection.y < 0)
		{
			t = t2;
			hit.intersection = o + t * d;
			if (t < 0 || hit.intersection.y > 1 || hit.intersection.y < 0)
			{
				return hit;
			}
		};

		hit.normal = glm::vec3(hit.intersection.x, -hit.intersection.y, hit.intersection.z);
		hit.normal = glm::normalize(hit.normal);

		Ray new_ray(o, d);
		Hit hit_plane = plane->intersect(new_ray);
		if (hit_plane.hit && hit_plane.distance < t && length(hit_plane.intersection - glm::vec3(0, 1, 0)) <= 1.0)
		{
			hit.intersection = hit_plane.intersection;
			hit.normal = hit_plane.normal;
		}

		hit.hit = true;
		hit.object = this;
		hit.intersection = transformationMatrix * glm::vec4(hit.intersection, 1.0); // implicit cast to vec3
		hit.normal = (normalMatrix * glm::vec4(hit.normal, 0.0));					// implicit cast to vec3
		hit.normal = glm::normalize(hit.normal);
		hit.distance = glm::length(hit.intersection - ray.origin);

		return hit;
	}
};

/**
 * @class Light.
 * @brief Represents a light in the scene.
 *
 * A light is represented by a position and a color.
 */
class Light
{
public:
	/**
	 * Position of the light source.
	 */
	glm::vec3 position;

	/**
	 * Color or intensity of the light source.
	 */
	glm::vec3 color;

	/**
	 * Constructor of the light.
	 *
	 * @param position Position of the light source
	 */
	Light(glm::vec3 position) : position(position)
	{
		color = glm::vec3(1.0);
	}

	/**
	 * Constructor of the light.
	 *
	 * @param position Position of the light source
	 * @param color Color or intensity of the light source
	 */
	Light(glm::vec3 position, glm::vec3 color) : position(position), color(color)
	{
	}
};

vector<Light *> lights; ///< A list of lights in the scene
glm::vec3 ambient_light(0.001, 0.001, 0.001);
vector<Object *> objects; ///< A list of all objects in the scene

glm::vec3 trace_ray(Ray ray);

/**
 * Function for computing color of an object according to the Phong Model.
 *
 * @param point A point belonging to the object for which the color is computer
 * @param normal A normal vector the the point
 * @param uv Texture coordinates
 * @param view_direction A normalized direction from the point to the viewer/camera
 * @param material A material structure representing the material of the object
 *
 * @return The color of the object in that point of intersection
 */
glm::vec3 PhongModel(glm::vec3 point, glm::vec3 normal, glm::vec2 uv, glm::vec3 view_direction, Material material)
{

	glm::vec3 color(0.0);
	for (int light_num = 0; light_num < lights.size(); light_num++)
	{

		glm::vec3 light_direction = glm::normalize(lights[light_num]->position - point);
		glm::vec3 reflected_direction = glm::reflect(-light_direction, normal);

		float NdotL = glm::clamp(glm::dot(normal, light_direction), 0.0f, 1.0f);
		float VdotR = glm::clamp(glm::dot(view_direction, reflected_direction), 0.0f, 1.0f);

		glm::vec3 diffuse_color = material.diffuse;
		if (material.texture)
		{
			diffuse_color = material.texture(uv);
		}

		glm::vec3 diffuse = diffuse_color * glm::vec3(NdotL);
		glm::vec3 specular = material.specular * glm::vec3(pow(VdotR, material.shininess));

		// Distance to the light
		float r = glm::distance(point, lights[light_num]->position);
		r = max(r, 0.1f);

		float shadow = 1.0;
		
		// To avoid noise
		glm::vec3 new_intersection_point = point + 0.001f * light_direction; 
		Ray inverse_light_ray(new_intersection_point, light_direction);

		for (int k = 0; k < objects.size(); k++)
		{
			Hit object_hit = objects[k]->intersect(inverse_light_ray);
			if (object_hit.hit && object_hit.distance < r)
			{
				shadow = 0.0;
				break;
			}
		}

		color += lights[light_num]->color * shadow * (diffuse + specular) / r / r;
	}
	color += ambient_light * material.ambient;

	color = glm::clamp(color, glm::vec3(0.0), glm::vec3(1.0));

	return color;
}

/**
 * Computes the closest intersection of a ray with an object in the scene.
 *
 * @param ray Ray that should be traced through the scene
 *
 * @return The intersection of the ray with the closest object in the scene
 */
Hit find_closest_hit(Ray ray)
{

	Hit closest_hit;

	closest_hit.hit = false;
	closest_hit.distance = INFINITY;

	for (int k = 0; k < objects.size(); k++)
	{
		Hit hit = objects[k]->intersect(ray);
		if (hit.hit == true && hit.distance < closest_hit.distance)
			closest_hit = hit;
	}

	return closest_hit;
}

/**
 * Computes a color along the ray.
 *
 * @param ray Ray that should be traced through the scene
 *
 * @return Color at the intersection point
 */
glm::vec3 trace_ray(Ray ray)
{

	bool inside_object = false;

	Hit closest_hit = find_closest_hit(ray);

	glm::vec3 color(0.0);

	if (closest_hit.hit)
	{

		Material object_material = closest_hit.object->getMaterial();

		if (object_material.reflectivity)
		{

			glm::vec3 reflection_direction = glm::reflect(ray.direction, closest_hit.normal);
			glm::vec3 new_intersection = closest_hit.intersection + 0.0001f * reflection_direction;
			Ray reflected_ray(new_intersection, reflection_direction);

			closest_hit = find_closest_hit(reflected_ray);
		}
		else
		{
			if (object_material.refraction)
			{

				float refractive_index = object_material.refractive_index;
				float eta = 1.0 / refractive_index; // air to glass

				glm::vec3 refraction_direction = glm::refract(ray.direction, closest_hit.normal, eta);
				glm::vec3 new_intersection = closest_hit.intersection + 0.001f * refraction_direction;
				Ray refracted_ray(new_intersection, refraction_direction);

				closest_hit = find_closest_hit(refracted_ray);

				eta = refractive_index / 1.0; // glass to ray

				refraction_direction = glm::refract(refraction_direction, -closest_hit.normal, eta);
				new_intersection = closest_hit.intersection + 0.001f * refraction_direction;
				Ray refracted_ray_2(new_intersection, refraction_direction);

				closest_hit = find_closest_hit(refracted_ray_2);
			}
		}
		if (closest_hit.hit)
			color = PhongModel(closest_hit.intersection, closest_hit.normal, closest_hit.uv, glm::normalize(-ray.direction), closest_hit.object->getMaterial());
		else
			color = glm::vec3(0.0);
	}
	else
	{
		color = glm::vec3(0.0, 0.0, 0.0);
	}
	return color;
}
/**
 * Scene definition.
 * This is the place to define all scene objects (objects, lights, materials, etc) and place them in the scene.
 */
void sceneDefinition()
{

	// Materials
	Material green_diffuse;
	green_diffuse.ambient = glm::vec3(0.03f, 0.1f, 0.03f);
	green_diffuse.diffuse = glm::vec3(0.3f, 1.0f, 0.3f);

	Material red_specular;
	red_specular.diffuse = glm::vec3(1.0f, 0.2f, 0.2f);
	red_specular.ambient = glm::vec3(0.01f, 0.02f, 0.02f);
	red_specular.specular = glm::vec3(0.5);
	red_specular.shininess = 10.0;

	Material blue_specular;
	blue_specular.ambient = glm::vec3(0.02f, 0.02f, 0.1f);
	blue_specular.diffuse = glm::vec3(0.2f, 0.2f, 1.0f);
	blue_specular.specular = glm::vec3(0.6);
	blue_specular.shininess = 100.0;

	Material reflective;
	reflective.reflectivity = true;

	Material refractive;
	refractive.refraction = true;
	refractive.refractive_index = 1.5f;

	Material textured;
	textured.texture = &rainbowTexture;

	Material red_diffuse;
	red_diffuse.ambient = glm::vec3(0.09f, 0.06f, 0.06f);
	red_diffuse.diffuse = glm::vec3(0.9f, 0.6f, 0.6f);

	Material blue_diffuse;
	blue_diffuse.ambient = glm::vec3(0.06f, 0.06f, 0.09f);
	blue_diffuse.diffuse = glm::vec3(0.6f, 0.6f, 0.9f);

	Material yellow_specular;
	yellow_specular.ambient = glm::vec3(0.1f, 0.10f, 0.0f);
	yellow_specular.diffuse = glm::vec3(0.4f, 0.4f, 0.0f);
	yellow_specular.specular = glm::vec3(1.0);
	yellow_specular.shininess = 100.0;

	// Cones
	Cone *cone = new Cone(yellow_specular);
	glm::mat4 translationMatrix = glm::translate(glm::vec3(5, 9, 14));
	glm::mat4 scalingMatrix = glm::scale(glm::vec3(3.0f, 12.0f, 3.0f));
	glm::mat4 rotationMatrix = glm::rotate(glm::radians(180.0f), glm::vec3(1, 0, 0));
	cone->setTransformation(translationMatrix * scalingMatrix * rotationMatrix);

	Cone *cone2 = new Cone(green_diffuse);
	translationMatrix = glm::translate(glm::vec3(6, -3, 7));
	scalingMatrix = glm::scale(glm::vec3(1.0f, 3.0f, 1.0f));
	rotationMatrix = glm::rotate(glm::atan(3.0f), glm::vec3(0, 0, 1));
	cone2->setTransformation(translationMatrix * rotationMatrix * scalingMatrix);

	objects.push_back(cone);
	objects.push_back(cone2);

	// Spheres
	objects.push_back(new Sphere(7.0, glm::vec3(-6, 4, 23), textured));
	objects.push_back(new Sphere(2.0, glm::vec3(-3, -1, 8), refractive));
	objects.push_back(new Sphere(1.0, glm::vec3(1, -2, 8), reflective));
	objects.push_back(new Sphere(0.5, glm::vec3(-1, -2.5, 6), red_specular));

	// Lights
	lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(1.0, 1.0, 1.0)));
	lights.push_back(new Light(glm::vec3(0, 1, 12), glm::vec3(0.1)));
	lights.push_back(new Light(glm::vec3(0, 5, 1), glm::vec3(0.4)));

	// Planes
	objects.push_back(new Plane(glm::vec3(0, -3, 0), glm::vec3(0.0, 1, 0)));
	objects.push_back(new Plane(glm::vec3(0, 1, 30), glm::vec3(0.0, 0.0, -1.0), green_diffuse));
	objects.push_back(new Plane(glm::vec3(-15, 1, 0), glm::vec3(1.0, 0.0, 0.0), red_diffuse));
	objects.push_back(new Plane(glm::vec3(15, 1, 0), glm::vec3(-1.0, 0.0, 0.0), blue_diffuse));
	objects.push_back(new Plane(glm::vec3(0, 27, 0), glm::vec3(0.0, -1, 0)));
	objects.push_back(new Plane(glm::vec3(0, 1, -0.01), glm::vec3(0.0, 0.0, 1.0), green_diffuse));
}

/**
 * Performs tonemapping of the intensities computed using the raytracer.
 *
 * @param intensity Input intensity
 *
 * @return Tonemapped intensity in range (0,1)
 */
glm::vec3 toneMapping(glm::vec3 intensity)
{
	float gamma = 1.0 / 2.0;
	float alpha = 12.0f;
	return glm::clamp(alpha * glm::pow(intensity, glm::vec3(gamma)), glm::vec3(0.0), glm::vec3(1.0));
}

/**
 * Main function.
 */
int main(int argc, const char *argv[])
{

	clock_t t = clock(); // variable for keeping the time of the rendering

	int width = 1024; // width of the image
	int height = 768; // height of the image
	float fov = 90;	  // field of view

	sceneDefinition(); // Let's define a scene

	Image image(width, height); // Create an image where we will store the result

	float s = 2 * tan(0.5 * fov / 180 * M_PI) / width;
	float X = -s * width / 2;
	float Y = s * height / 2;

	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{

			float dx = X + i * s + s / 2;
			float dy = Y - j * s - s / 2;
			float dz = 1;

			glm::vec3 origin(0, 0, 0);
			glm::vec3 direction(dx, dy, dz);
			direction = glm::normalize(direction);

			Ray ray(origin, direction);

			image.setPixel(i, j, toneMapping(trace_ray(ray)));
		}

	t = clock() - t;
	cout << "It took " << ((float)t) / CLOCKS_PER_SEC << " seconds to render the image." << endl;
	cout << "I could render at " << (float)CLOCKS_PER_SEC / ((float)t) << " frames per second." << endl;

	// Writing the final results of the rendering
	if (argc == 2)
	{
		image.writeImage(argv[1]);
	}
	else
	{
		image.writeImage("./result.ppm");
	}

	return 0;
}
