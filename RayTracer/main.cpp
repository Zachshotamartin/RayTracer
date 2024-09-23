#include "headers_and_constants.h"


int main() {
    
    hittable_list world;
    vec3 u = vec3(1, 0, 0);
    vec3 v = vec3(0, 1, 0);
    vec3 w = vec3(0, 0, 1);
    vec3 cube_side_lengths = vec3(.5, .5, .5);
    
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
        world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = random_double();
                auto choose_shape = random_double();
                point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    shared_ptr<material> material;
                    
                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = color::random() * color::random();
                        material = make_shared<lambertian>(albedo);
                        if (choose_shape < .5) { // sphere
                            world.add(make_shared<sphere>(center, 0.2, material));
                        }
                        else {
                            world.add(make_shared<cube>(center, cube_side_lengths, u, v, w, material));
                        }
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = color::random(0.5, 1);
                        auto fuzz = random_double(0, 0.5);
                        material = make_shared<metal>(albedo, fuzz);
                        if (choose_shape < .5) { // sphere
                            world.add(make_shared<sphere>(center, 0.2, material));
                        }
                        else {
                            world.add(make_shared<cube>(center, cube_side_lengths, u, v, w, material));
                        }
                    } else {
                        // glass
                        material = make_shared<dielectric>(1.5);
                        if (choose_shape < .5) { //sphere
                            world.add(make_shared<sphere>(center, 0.2, material));
                        }
                        else {
                            world.add(make_shared<cube>(center, cube_side_lengths, u, v, w, material));
                        }
                    }
                }
            }
        }

        auto material1 = make_shared<dielectric>(1.5);
        world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

        auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
        world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

        auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
        world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

        camera cam;

        cam.aspect_ratio      = 16.0 / 9.0;
        cam.image_width       = 600;
        cam.samples_per_pixel = 10;
        cam.max_depth         = 10;

        cam.vfov     = 20;
        cam.lookfrom = point3(13,2,3);
        cam.lookat   = point3(0,0,0);
        cam.vup      = vec3(0,1,0);

        cam.defocus_angle = 0.6;
        cam.focus_dist    = 10.0;
    
    cam.display(world);
}
