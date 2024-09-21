#ifndef SPHERE_H
#define SPHERE_H


class sphere : public hittable {
  public:
    sphere(const point3& center, double radius) : center(center), radius(std::fmax(0,radius)) {}

    bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override {
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
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (b + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;

        return true;
    }

  private:
    point3 center;
    double radius;
};

#endif
