"""spot_pico tibia four-bar linkage: servo horn <-> URDF tibia joint mapping.

Geometry measured from CAD (leg:1, all legs identical):
  crank r1 = 22.22 mm  (servo horn, coaxial with femur joint)
  rod   L  = 50.13 mm  (ball-link pushrod)
  lever r2 = 13.01 mm  (tibia-side lever about the knee)
  ground g = 57.92 mm  (femur: hip-of-femur to knee)

The four-bar lives in the FEMUR frame: its input is the crank angle RELATIVE
TO THE FEMUR. Since the servo is mounted in the coxa, that input equals
(servo_horn_angle - q_femur), both measured relative to the coxa.

Conventions (all radians):
  q_tibia  : URDF tibia joint angle, 0 = CAD stance pose
  q_femur  : URDF femur joint angle, 0 = CAD stance pose
  q_servo  : servo horn angle relative to coxa, 0 = CAD stance pose
Mirrored legs (fl, rl) use mirrored URDF axes, so the same scalar map applies
in joint coordinates. Verify signs once on hardware.
"""
import math

R1, L, R2, G = 2.2222, 5.0125, 1.3006, 5.7925   # cm
PHI0 = math.radians(65.31)    # crank angle at stance, from femur line
PSI0 = math.radians(85.68)    # lever angle at stance, from femur line

def _lever_from_crank(phi):
    """four-bar: crank angle -> lever angle (femur-frame absolute angles)."""
    b1y, b1z = R1*math.cos(phi), R1*math.sin(phi)
    dy, dz = b1y - G, b1z
    dist = math.hypot(dy, dz)
    c = (dist*dist + R2*R2 - L*L) / (2.0*dist*R2)
    if abs(c) > 1.0:
        raise ValueError("linkage cannot close: crank angle out of range")
    a = math.atan2(dz, dy)
    c1, c2 = a + math.acos(c), a - math.acos(c)
    return min((c1, c2), key=lambda x: abs(_wrap(x - PSI0)))

def _crank_from_lever(psi):
    """inverse: lever angle -> crank angle."""
    b2y, b2z = G + R2*math.cos(psi), R2*math.sin(psi)
    dist = math.hypot(b2y, b2z)
    c = (dist*dist + R1*R1 - L*L) / (2.0*dist*R1)
    if abs(c) > 1.0:
        raise ValueError("linkage cannot close: tibia angle out of range")
    a = math.atan2(b2z, b2y)
    # two candidates; pick the one nearest the stance crank angle's branch
    c1, c2 = a + math.acos(c), a - math.acos(c)
    return min((c1, c2), key=lambda x: abs(_wrap(x - PHI0)))

def _wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def tibia_from_servo(q_servo, q_femur=0.0):
    """URDF tibia joint angle produced by a given servo horn angle."""
    return _wrap(_lever_from_crank(PHI0 + (q_servo - q_femur)) - PSI0)

def servo_from_tibia(q_tibia, q_femur=0.0):
    """Servo horn angle required for a desired URDF tibia joint angle.
    This is what your low-level controller needs."""
    return _wrap(_crank_from_lever(PSI0 + q_tibia) - PHI0) + q_femur

def ratio(q_tibia=0.0, q_femur=0.0):
    """instantaneous dq_tibia/dq_servo at a pose (~1.64 at stance)."""
    h = 1e-5
    s = servo_from_tibia(q_tibia, q_femur)
    return 2*h / (servo_from_tibia(q_tibia+h, q_femur) - servo_from_tibia(q_tibia-h, q_femur)) * (2*h) / (2*h)

if __name__ == "__main__":
    d = math.degrees; r = math.radians
    print("self-test: round-trip and ratio")
    for t_deg in (-60, -30, 0, 30, 55):
        s = servo_from_tibia(r(t_deg))
        back = tibia_from_servo(s)
        rt = ratio(r(t_deg))
        print(f"  tibia {t_deg:+4d} deg -> servo {d(s):+7.2f} deg  (roundtrip {d(back):+7.2f}, ratio {rt:.3f})")
    # femur coupling: tibia held constant while femur moves requires servo to follow
    s0 = servo_from_tibia(0.0, 0.0); s1 = servo_from_tibia(0.0, r(20))
    print(f"  femur +20 deg with tibia fixed: servo must move {d(s1-s0):+.2f} deg (coupling = 1:1 with femur)")
