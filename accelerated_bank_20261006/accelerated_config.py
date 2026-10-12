import json
from pathlib import Path
CONFIG=json.loads((Path(__file__).resolve().parent.parent/'accelerated_research_20261006/protocol.json').read_text())
def configure(protocol,trainer,baseline,updates):
 protocol['protocol_id']=CONFIG['protocol_id']
 protocol['training']['batch_size']=CONFIG['training_batch']
 protocol['training']['policy_updates']=updates
 trainer['validation_batch_size']=CONFIG['validation_batch']
 trainer['validation_interval_updates']=CONFIG['validation_interval']
 trainer['checkpoint_interval_updates']=CONFIG['validation_interval']
 trainer['structured_log_interval_updates']=1
 baseline['search']['score_microbatch']=CONFIG['score_microbatch']
