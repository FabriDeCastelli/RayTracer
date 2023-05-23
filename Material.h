/**
 * @file Material.h
 */

//
//  Material.h
//  Raytracer
//

#ifndef Material_h
#define Material_h

#include "glm/glm.hpp"
#include "Textures.h"

/**
 * @struct Material.
 * @brief Represents the material of an object.
 *
 * The material is represented by its ambient, diffuse and specular colors,
 * its shininess, and whether it is reflective or refractive.
 * It can also have a texture.
 */
struct Material {

    /**
     * Ambient color.
     */
    glm::vec3 ambient = glm::vec3(0.0);

    /**
     * Diffuse color.
     */
    glm::vec3 diffuse = glm::vec3(1.0);

    /**
     * Specular color.
     */
    glm::vec3 specular = glm::vec3(0.0);

    /**
     * Whether the material is reflective.
     */
    bool reflectivity = false;

    /**
     * Whether the material is refractive.
     */
    bool refraction = false;

    /**
     * Refractive index.
     */
    float refractive_index = 0.0;

    /**
     * Shininess.
     */
    float shininess = 0.0;

    /**
     * The texture function.
     */
    glm::vec3 (*texture)(glm::vec2 uv) = NULL;
};

#endif /* Material_h */
