import jax.numpy as jnp

from dataclasses import dataclass, field


@dataclass
class TrainingEnvironment:
    """
    A single training environment with goals and distractors.

    Goals are objects the agent is rewarded for reaching.
    Distractors are objects present in the environment but not rewarded.
    """
    goals: list[jnp.ndarray] = field(default_factory=list)
    distractors: list[jnp.ndarray] = field(default_factory=list)

    @property
    def n_goals(self) -> int:
        return len(self.goals)

    @property
    def n_distractors(self) -> int:
        return len(self.distractors)

    @property
    def all_objects(self) -> list[jnp.ndarray]:
        """All objects in the environment (goals + distractors)"""
        return self.goals + self.distractors

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON caching."""
        return {
            "goals": [g.tolist() for g in self.goals],
            "distractors": [d.tolist() for d in self.distractors],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingEnvironment":
        """Deserialize from dictionary."""
        return cls(
            goals=[jnp.array(g) for g in data["goals"]],
            distractors=[jnp.array(d) for d in data["distractors"]],
        )


@dataclass
class TrainingStage:
    """
    A training stage: set of environments trained on with averaged gradients.

    Environments within a stage are trained on simultaneously but separately,
    meaning the agent alternates between them (or sees them with equal probability)
    and gradient updates are averaged.
    """
    environments: list[TrainingEnvironment] = field(default_factory=list)

    @property
    def n_environments(self) -> int:
        return len(self.environments)

    def get_all_goal_features(self) -> jnp.ndarray:
        """
        Get combined goal features for simple models that lump features together.
        Returns binary vector where 1 indicates feature is present in any goal.
        """
        if not self.environments:
            raise ValueError("Stage has no environments")

        n_features = self.environments[0].goals[0].shape[0]
        combined = jnp.zeros(n_features)

        for env in self.environments:
            for goal in env.goals:
                combined = jnp.maximum(combined, goal)

        return combined

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON caching."""
        return {"environments": [e.to_dict() for e in self.environments]}

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingStage":
        """Deserialize from dictionary."""
        return cls(environments=[TrainingEnvironment.from_dict(e) for e in data["environments"]])


@dataclass
class TrainingPipeline:
    """
    Complete training pipeline as sequence of stages.

    Stages are processed sequentially (separated by _then_ in run names),
    with each stage reaching equilibrium before the next begins.
    """
    stages: list[TrainingStage] = field(default_factory=list)

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:
        lines = [f"TrainingPipeline with {self.n_stages} stages:"]
        for i, stage in enumerate(self.stages):
            lines.append(f"  Stage {i+1}: {stage.n_environments} environment(s)")
            for j, env in enumerate(stage.environments):
                goal_str = ", ".join([f"goal_{k}" for k in range(env.n_goals)])
                dist_str = ", ".join([f"dist_{k}" for k in range(env.n_distractors)])
                lines.append(f"    Env {j+1}: goals=[{goal_str}], distractors=[{dist_str}]")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON caching."""
        return {"stages": [s.to_dict() for s in self.stages]}

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingPipeline":
        """Deserialize from dictionary."""
        return cls(stages=[TrainingStage.from_dict(s) for s in data["stages"]])


@dataclass
class PaddedPipeline:
    """
    Padded pipeline representation for batched JAX processing.

    All arrays are padded to max sizes with corresponding masks indicating valid entries.
    """
    # Shape: (max_stages, max_envs_per_stage, max_goals_per_env, n_features)
    goals: jnp.ndarray
    # Shape: (max_stages, max_envs_per_stage, max_distractors_per_env, n_features)
    distractors: jnp.ndarray
    # Shape: (max_stages,) - which stages are valid
    stage_mask: jnp.ndarray
    # Shape: (max_stages, max_envs_per_stage) - which envs are valid per stage
    env_mask: jnp.ndarray
    # Shape: (max_stages, max_envs_per_stage, max_goals_per_env) - which goals are valid
    goal_mask: jnp.ndarray
    # Shape: (max_stages, max_envs_per_stage, max_distractors_per_env) - which distractors are valid
    distractor_mask: jnp.ndarray

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON caching."""
        return {
            "goals": self.goals.tolist(),
            "distractors": self.distractors.tolist(),
            "stage_mask": self.stage_mask.tolist(),
            "env_mask": self.env_mask.tolist(),
            "goal_mask": self.goal_mask.tolist(),
            "distractor_mask": self.distractor_mask.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaddedPipeline":
        """Deserialize from dictionary."""
        return cls(
            goals=jnp.array(data["goals"]),
            distractors=jnp.array(data["distractors"]),
            stage_mask=jnp.array(data["stage_mask"]),
            env_mask=jnp.array(data["env_mask"]),
            goal_mask=jnp.array(data["goal_mask"]),
            distractor_mask=jnp.array(data["distractor_mask"]),
        )

