#ifndef SPHERE_H
#define SPHERE_H


class sphere : public hittable {
  public:
    sphere(const point3& center, double radius, shared_ptr<material> mat)
          : center(center), radius(std::fmax(0,radius)), mat(mat) {}

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        vec3 oc = center - r.o();
        auto a = r.d().dot();
        auto b = dot(r.d(), oc);
        auto c = oc.dot() - radius*radius;

        auto discr = discriminant(a, b, c);
        if (discr < 0)
            return false;

        auto sqrtd = std::sqrt(discr);

        // Find the nearest root that lies in the acceptable range.
                auto root = (b - sqrtd) / a;
                if (!ray_t.surrounds(root)) {
                    root = (b + sqrtd) / a;
                    if (!ray_t.surrounds(root))
                        return false;
                }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;
        return true;
    }

  private:
    point3 center;
    double radius;
    shared_ptr<material> mat;
};

#endif
