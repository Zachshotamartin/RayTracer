#include "integrator.h"
#include "caustics.h"
namespace {
double power_weight(double a, double b) {
    if (a <= 0)
        return 0;
    double ratio = b / a;
    return 1 / (1 + ratio * ratio);
}
} // namespace
color trace_path(ray r, const hittable &world, const light_list &lights, const environment &env,
                 int max_depth, sampler &rng, ray_counts &counts, transport_options options,
                 hit_record *primary_hit, bool *primary_valid) {
    if (primary_valid)
        *primary_valid = false;
    color radiance, throughput(1, 1, 1);
    bool previous_delta = true;
    double previous_pdf = 0;
    point3 previous_origin = r.o();
    bool diffuse_candidate = false, shadow_candidate = false, has_specular = false,
         transmission_only = true;
    for (int bounce = 0; bounce < max_depth; ++bounce) {
        ++counts.paths;
        hit_record rec;
        if (!world.hit(r, interval(1e-8, infinity), rec)) {
            radiance += throughput * env.radiance(r.d());
            break;
        }
        if (bounce == 0) {
            if (primary_hit)
                *primary_hit = rec;
            if (primary_valid)
                *primary_valid = true;
        }
        color emission = rec.mat->emitted(rec);
        double weight = previous_delta
                            ? 1
                            : power_weight(previous_pdf, lights.pdf_for_hit(previous_origin, rec));
        bool replaced = has_specular && lights.owns_emitter(rec.object) &&
                        ((options.caustics && diffuse_candidate) ||
                         (options.transparent_shadows && shadow_candidate && transmission_only));
        if (!replaced)
            radiance += throughput * emission * weight;
        if (options.caustics)
            radiance += throughput * options.caustics->radiance(r.d(), rec, max_depth - bounce - 1);

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
                    bool crossed_glass = false;
                    color visibility =
                        options.transparent_shadows
                            ? light_list::transmittance(origin, direct, world, &crossed_glass)
                            : (light_list::visible(origin, direct, world) ? color(1, 1, 1)
                                                                          : color{});
                    if (!visibility.near_zero()) {
                        double mis = (direct.delta || crossed_glass || bounce + 1 == max_depth)
                                         ? 1
                                         : power_weight(direct.pdf,
                                                        rec.mat->pdf(r.d(), direct.direction, rec));
                        radiance += throughput * bsdf * direct.radiance * visibility *
                                    (cosine * mis / direct.pdf);
                    }
                }
            }
        }
        if (bounce + 1 == max_depth)
            break;
        scatter_sample scattered;
        if (!rec.mat->sample(r.d(), rec, rng, scattered))
            break;
        if (!scattered.delta) {
            diffuse_candidate = rec.mat->is_diffuse();
            shadow_candidate = true;
            has_specular = false;
            transmission_only = true;
        } else {
            has_specular = true;
            transmission_only = transmission_only && scattered.transmitted;
        }
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
