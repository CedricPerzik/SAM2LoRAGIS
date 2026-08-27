'''
python -c "
import optuna
s = optuna.load_study(study_name='hp_sweep', storage='sqlite:////home/s2145588/thesis/sam2loraboracluster/hp_study.db')
print(s.best_params)
print(1 - s.best_value)
"
'''
# After 9 trials: 
'''
     'base_lr': 5.745509983517318e-05
   'lora_rank': 16
'tversky_beta': 0.6495571120940955
'weight_decay': 0.2757032604707855

      1-dice? : 0.983365869142984
'''

# After 17 trials:
'''
     'base_lr': 5.442391344622082e-05
   'lora_rank': 16
'tversky_beta': 0.5572054074111565
'weight_decay': 0.0015760723171026822

      1-dice? : 0.9837653771727329
'''

#%%
import optuna
from optuna.importance import FanovaImportanceEvaluator
optuna.logging.set_verbosity(optuna.logging.WARNING)

s = optuna.load_study(
    study_name="hp_sweep",
    storage="sqlite:////home/ced/Documents/unicluster/outputs/databases/hp_study_batch3.db"
)

# Filter out degenerate trials (objective == 1.0 means val_dice == 0 — training collapsed)
good_trials = [t for t in s.trials if t.value is not None and t.value < 1.0]
fs = optuna.create_study(direction=s.direction)
for t in good_trials:
    fs.add_trial(t)

print(f"Using {len(good_trials)} / {len(s.trials)} trials (excluded {len(s.trials) - len(good_trials)} with value=1.0)")

direction_label = "lower is better" if fs.direction.name == "MINIMIZE" else "higher is better"
obj_label = f"Objective (1 - dice)  [{direction_label}]"

# %%
# optimization history — did results improve over trials?
fig = optuna.visualization.plot_optimization_history(fs)
fig.update_yaxes(title_text=obj_label)
fig.show()
# %%
# which parameters mattered most? (needs lots of trials to be meaningful)
'''
plot_param_importances runs fANOVA (functional ANOVA), which fits a random forest on (hyperparameter → objective) data 
to estimate each parameter's influence. Random forests have inherent randomness in their training 
(bootstrap sampling, random feature splits), and since no seed is fixed, each run produces a slightly different forest 
→ slightly different importance scores. Hence we have a fixed seed here.
'''
fig = optuna.visualization.plot_param_importances(fs, evaluator=FanovaImportanceEvaluator(seed=42))
fig.update_xaxes(title_text="Importance for objective (1 - dice)  [higher = more influential]")
fig.show()
# %%
# all trials as parallel coordinates — spot patterns
fig = optuna.visualization.plot_parallel_coordinate(fs)
fig.update_traces(line_colorbar_title_text=obj_label)
fig.show()
# %%
# each parameter vs objective — see the landscape
fig = optuna.visualization.plot_slice(fs)
fig.update_yaxes(title_text=obj_label)
fig.show()

# %%
# best trial summary
best = fs.best_trial
print(f"Best trial: #{best.number}")
print(f"  Objective (1 - dice): {best.value:.6f}  ->  val_dice: {1 - best.value:.6f}")
print("  Params:")
for k, v in best.params.items():
    print(f"    {k}: {v}")
# %%
'''
batch 2:
Best trial: #13
  Objective (1 - dice): 0.016235  ->  val_dice: 0.983765
  Params:
    base_lr: 5.442391344622082e-05
    lora_rank: 16
    tversky_beta: 0.5572054074111565
    weight_decay: 0.0015760723171026822

batch 3:
Best trial: #20
  Objective (1 - dice): 0.016096  ->  val_dice: 0.983904
  Params:
    base_lr: 7.102526614544882e-05
    lora_rank: 16
    tversky_beta: 0.5513642989345549
    weight_decay: 0.0035071886076701955
'''

# %%
