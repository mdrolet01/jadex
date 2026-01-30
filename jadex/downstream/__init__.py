def register_downstream():
    from jadex.downstream.image_label.models import register_image_label_df_models
    from jadex.downstream.traj_gpt.models import register_traj_gpt_models

    register_image_label_df_models()
    register_traj_gpt_models()
