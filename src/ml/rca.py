from typing import List, Dict
def suggest_rca(ncr: Dict, similar: List[Dict]) -> Dict:
    # Simple heuristic RCA builder:
    causes = []
    contributing = []
    actions = []
    for s in similar:
        dt = s.get('defect_type')
        if dt:
            causes.append(f"Historical defect: {dt}")
    # Basic heuristics from keywords
    txt = (ncr.get('title','') + ' ' + ncr.get('description','')).lower()
    if 'leak' in txt or 'seam' in txt:
        causes.append('Seaming tooling or seam quality')
        actions.append('Inspect seamer tooling; verify seam specs; sample seam cross-section')
    if 'underfill' in txt or 'under weight' in txt:
        causes.append('Filling calibration or intermittent nozzle fault')
        actions.append('Calibrate filler; inspect nozzles; review fill logs for spikes')
    return {
        'root_causes': list(dict.fromkeys(causes)),
        'contributing_factors': list(dict.fromkeys(contributing)),
        'recommended_actions': list(dict.fromkeys(actions)) or ['Run containment — segregate affected lots; investigate with 5-why']
    }