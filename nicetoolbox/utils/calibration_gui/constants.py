px = 10
py = 5


all_fields = dict(
    cams=dict(
        description="Camera name and image resolution (width x height)",
        variables=[dict(name="name", type="str"), dict(name="size", type="1x2")],
    ),
    K=dict(description="Intrinsic matrix K", variables=[dict(name="K", type="3x3")]),
    mtx=dict(description="Camera matrix mtx", variables=[dict(name="mtx", type="3x3")]),
    Rt=dict(
        description="Rotation matrix R and translation matrix t",
        variables=[dict(name="R", type="3x3"), dict(name="t", type="3x1")],
    ),
    rtvec=dict(
        description="Rotation vectors rvec (Rodrigues) and translation vector tvec",
        variables=[dict(name="rvec", type="1x3"), dict(name="tvec", type="3x1")],
    ),
    d=dict(
        description="Distortion coefficients d",
        variables=[dict(name="d", type="1x5")],  # (k1, k2, p1, p2, k3)
    ),
    dist=dict(
        description="Distortion coefficients dist",
        variables=[dict(name="dist", type="1x5")],  # (k1, k2, p1, p2, k3)
    ),
)


matrix_name_synonyms = [
    {"K", "mtx", "intrinsic_matrix"},
    {"R", "rotation_matrix"},
    {"t", "translation", "tvec", "tvecs"},
    {"rvec", "rvecs"},
    {"d", "dist", "distortions"},
    {"Rt", "extrinsics_matrix"},
    {"P", "projection_matrix"},
    {"name", "camera_name"},
    {"size", "image_size", "image_resolution", "resolution"},
]
