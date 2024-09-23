#ifndef cube_h
#define cube_h


class cube : public hittable {
public:
    

    cube(const vec3& center, const vec3& side_lengths, const vec3& u, const vec3& v, const vec3& w, shared_ptr<material> mat)
        : center(center), side_lengths(side_lengths), u(u), v(v), w(w), mat(mat) {}

    // Further methods to handle ray intersections, etc.
    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // Step 1: Transform ray to the cube's local space
        vec3 local_ray_origin = vec3(dot(r.o() - center, u), dot(r.o() - center, v), dot(r.o() - center, w));
        vec3 local_ray_direction = vec3(dot(r.d(), u), dot(r.d(), v), dot(r.d(), w));

        // Step 2: Define AABB bounds in local space (assuming cube is axis-aligned in local space)
        vec3 local_min = -side_lengths / 2.0;  // Half-lengths of each side
        vec3 local_max = side_lengths / 2.0;
        double t_min = ray_t.min;  // Start with the minimum value from the ray's valid range
        double t_max = ray_t.max;
        // Step 3: Check for ray-AABB intersection in local space
        for (int i = 0; i < 3; i++) {
            double invD = 1.0 / local_ray_direction[i];
            double t0 = (local_min[i] - local_ray_origin[i]) * invD;
            double t1 = (local_max[i] - local_ray_origin[i]) * invD;
            if (invD < 0.0) std::swap(t0, t1);
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min) return false;
        }

        // Step 4: Record hit data
        rec.t = t_min;  // The nearest intersection time
        rec.p = r.at(rec.t);  // Intersection point in world space

        // Step 5: Transform local normal back to world space
        vec3 local_normal;
        if (fabs(rec.p.x() - local_min.x()) < 1e-6) local_normal = vec3(-1, 0, 0);  // Left face
        else if (fabs(rec.p.x() - local_max.x()) < 1e-6) local_normal = vec3(1, 0, 0);  // Right face
        else if (fabs(rec.p.y() - local_min.y()) < 1e-6) local_normal = vec3(0, -1, 0);  // Bottom face
        else if (fabs(rec.p.y() - local_max.y()) < 1e-6) local_normal = vec3(0, 1, 0);  // Top face
        else if (fabs(rec.p.z() - local_min.z()) < 1e-6) local_normal = vec3(0, 0, -1);  // Back face
        else local_normal = vec3(0, 0, 1);  // Front face

        rec.normal = local_normal.x() * u + local_normal.y() * v + local_normal.z() * w;  // Transform to world space
        rec.set_face_normal(r, rec.normal);  // Flip normal if necessary
        rec.mat = mat;  // Assign the material
        return true;
    }

    
private:
    vec3 center;
    vec3 side_lengths; // Lengths along the local axes
    vec3 u, v, w;      // Local axes representing the cube's orientation
    shared_ptr<material> mat;
};
#endif
