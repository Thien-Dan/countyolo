"""Grant read-only access only to this project's configured Kaggle accounts."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from kaggle_watch import accounts,api_for

def share(ref,readers,apply=False):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest,ApiUpdateDatasetMetadataRequest
    from kagglesdk.datasets.types.dataset_types import DatasetSettings,DatasetCollaborator
    from kagglesdk.users.types.users_enums import CollaboratorType
    owner,slug=ref.split('/')
    keys=accounts()
    if owner not in keys or any(u not in keys for u in readers):
        raise ValueError('Only configured project accounts may receive access')
    api=api_for(owner,keys[owner])
    get=ApiGetDatasetMetadataRequest();get.owner_slug=owner;get.dataset_slug=slug
    with api.build_kaggle_client() as client:
        response=client.datasets.dataset_api_client.get_dataset_metadata(get)
    if response.error_message: raise RuntimeError(response.error_message)
    info=response.info
    if not info.is_private: raise RuntimeError('Expected a private input')
    folder=ROOT/'outputs/dataset_access'/ref.replace('/','__');folder.mkdir(parents=True,exist_ok=True)
    snapshot=folder/'metadata_before.json'
    if not snapshot.exists(): snapshot.write_text(response.to_json(info),encoding='utf-8')
    settings=DatasetSettings()
    for field in ('title','subtitle','description','is_private','keywords','licenses','collaborators','data','expected_update_frequency','user_specified_sources'):
        setattr(settings,field,getattr(info,field))
    existing={c.username for c in settings.collaborators}
    for user in readers:
        if user!=owner and user not in existing:
            c=DatasetCollaborator();c.username=user;c.role=CollaboratorType.READER
            settings.collaborators.append(c)
    print('Read-only access plan:',ref,'private=True',readers,flush=True)
    if not apply: return
    request=ApiUpdateDatasetMetadataRequest();request.owner_slug=owner;request.dataset_slug=slug;request.settings=settings
    with api.build_kaggle_client() as client:
        result=client.datasets.dataset_api_client.update_dataset_metadata(request)
        if result.errors: raise RuntimeError('; '.join(result.errors))
        after=client.datasets.dataset_api_client.get_dataset_metadata(get)
    assert after.info.is_private
    access={c.username:c.role.name for c in after.info.collaborators}
    assert all(u==owner or u in access for u in readers)
    for user in readers:
        reader=api_for(user,keys[user]);files=reader.dataset_list_files(ref)
        print('Verified read access:',user,ref,'files',len(files.files),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--readers',nargs='+',required=True);p.add_argument('--apply',action='store_true')
    a=p.parse_args();share(a.dataset,a.readers,a.apply)
