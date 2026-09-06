#include "integrator.h"
namespace {
double power_weight(double a, double b) {
    if (a <= 0)
        return 0;
    double ratio = b / a;
    return 1 / (1 + ratio * ratio);
}
} // namespace
color trace_path(ray r, const hittable &world, const light_list &lights, const environment &env,
                 int max_depth, sampler &rng, ray_counts &counts) {
    color radiance, throughput(1, 1, 1);
    bool previous_delta = true;
    double previous_pdf = 0;
    point3 previous_origin = r.o();
    for (int bounce = 0; bounce < max_depth; ++bounce) {
        ++counts.paths;
        hit_record rec;
        if (!world.hit(r, interval(1e-8, infinity), rec)) {
            radiance += throughput * env.radiance(r.d());
            break;
        }
        color emission = rec.mat->emitted(rec);
        double weight = previous_delta
                            ? 1
                            : power_weight(previous_pdf, lights.pdf_for_hit(previous_origin, rec));
        radiance += throughput * emission * weight;

        // Next-event estimation is additive and uses this material's BSDF.
        // MIS balances sampled area lights against paths that hit their geometry.
        if (!rec.mat->is_delta() && !lights.empty()) {
            point3 origin = offset_origin(rec, rec.normal);
            auto direct = lights.sample(origin, rng);
            double cosine = dot(rec.normal, direct.direction);
            if (direct.pdf > 0 && cosine > 0) {
                color bsdf = rec.mat->evaluate(r.d(), direct.direction, rec);
                if (!bsdf.near_zero()) {
                    ++counts.shadows;
                    if (light_list::visible(origin, direct, world)) {
                        double mis = (direct.delta || bounce + 1 == max_depth)
                                         ? 1
                                         : power_weight(direct.pdf,
                                                        rec.mat->pdf(r.d(), direct.direction, rec));
                        radiance +=
                            throughput * bsdf * direct.radiance * (cosine * mis / direct.pdf);
                    }
                }
            }
        }
        if (bounce + 1 == max_depth)
            break;
        scatter_sample scattered;
        if (!rec.mat->sample(r.d(), rec, rng, scattered))
            break;
        throughput = throughput * scattered.weight;
        if (throughput.near_zero())
            break;
        previous_delta = scattered.delta;
        previous_pdf = scattered.pdf;
        previous_origin = offset_origin(rec, scattered.direction);
        r = ray(previous_origin, to_unit_vector(scattered.direction));
    }
    return radiance;
}
