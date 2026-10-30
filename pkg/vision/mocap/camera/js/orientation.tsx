

// TODO: Generalize to sideways cameras.
// TODO: Also add this to the per-camera UI.
// Returns null for a camera without an accelerometer: callers leave its image
// unmirrored and show no angle.
export function camera_orientation(camera_status) {

    let accel = camera_status.accelerometer && camera_status.accelerometer.value;
    if (!accel || !accel.values || accel.values.length < 3) {
        return null;
    }

    let y = accel.values[1];
    let z = accel.values[2];

    let upside_down = y > 0 ? true : false;

    let horizon_angle = Math.round((180 / Math.PI) * Math.atan(z / Math.abs(y)));

    return {
        upside_down,
        horizon_angle
    };
}