import os
import numpy as np
import tqdm
import librosa
import torch
from scipy.spatial.transform import Rotation as R

from aist_plusplus.loader import AISTDataset


class AIChoreoDataset:
    def __init__(self, root_dir, audio_dir, audio_cache_dir="./audio_cache", split="train",
                 m_seq_len=120, a_seq_len=240, out_seq_len=20):
        self.root_dir = root_dir
        self.audio_dir = audio_dir
        self.audio_cache_dir = audio_cache_dir
        self.split = split
        self.m_seq_len = m_seq_len
        self.a_seq_len = a_seq_len
        self.out_seq_len = out_seq_len
        self.aist_dataset = AISTDataset(root_dir)

        self.seq_names = []
        if "train" in split:
            self.seq_names += np.loadtxt(
                os.path.join(root_dir, "splits/crossmodal_train.txt"), dtype=str).tolist()
        if "val" in split:
            self.seq_names += np.loadtxt(
                os.path.join(root_dir, "splits/crossmodal_val.txt"), dtype=str).tolist()
        if "test" in split:
            self.seq_names += np.loadtxt(
                os.path.join(root_dir, "splits/crossmodal_test.txt"), dtype=str).tolist()
        ignore_list = np.loadtxt(os.path.join(root_dir, "ignore_list.txt"), dtype=str).tolist()
        self.seq_names = [n for n in self.seq_names if n not in ignore_list ]

        print ("Pre-compute audio features on the entire (train + val + test) dataset.")
        os.makedirs(audio_cache_dir, exist_ok=True)
        self.cache_audio_features()

        # self replication
        if split == "train":
            self.seq_names = self.seq_names * 100

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, index):
        # load
        seq_name = self.seq_names[index]
        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
            self.aist_dataset.motion_dir, seq_name)
        smpl_trans /= smpl_scaling
        audio = self.load_cached_audio_features(seq_name)

        start = np.random.randint(
            0, min(smpl_poses.shape[0] - self.m_seq_len - self.out_seq_len, audio.shape[0] - self.a_seq_len))
        motion_end = start + self.m_seq_len + self.out_seq_len
        audio_end = start + self.a_seq_len

        smpl_poses = smpl_poses[start:motion_end]
        smpl_trans = smpl_trans[start:motion_end]
        smpl_poses = R.from_rotvec(
            smpl_poses.reshape(-1, 3)).as_matrix().reshape(smpl_poses.shape[0], -1)
        smpl_motion = np.concatenate([smpl_poses, smpl_trans], axis=-1)
        assert smpl_motion.shape[-1] == 24 * 9 + 3, f"motion shape is {smpl_motion.shape}!"
        motion = smpl_motion[:self.m_seq_len]
        target = smpl_motion[-self.out_seq_len:]

        audio = audio[start:audio_end]

        return (
            torch.from_numpy(motion).float(),
            torch.from_numpy(audio).float(),
            torch.from_numpy(target).float(),
            seq_name
        )

    def load_cached_audio_features(self, seq_name):
        audio_name = seq_name.split("_")[-2]
        return np.load(os.path.join(self.audio_cache_dir, f"{audio_name}.npy"))

    def cache_audio_features(self):
        FPS = 60
        HOP_LENGTH = 512
        SR = FPS * HOP_LENGTH
        EPS = 1e-6

        def _get_tempo(audio_name):
            """Get tempo (BPM) for a music by parsing music name."""
            assert len(audio_name) == 4
            if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
                return int(audio_name[3]) * 10 + 80
            elif audio_name[0:3] == 'mHO':
                return int(audio_name[3]) * 5 + 110
            else: assert False, audio_name

        seq_names = self.aist_dataset.mapping_seq2env.keys()
        audio_names = list(set([seq_name.split("_")[-2] for seq_name in seq_names]))

        for audio_name in tqdm.tqdm(audio_names):
            save_path = os.path.join(self.audio_cache_dir, f"{audio_name}.npy")
            if os.path.exists(save_path):
                continue
            data, _ = librosa.load(os.path.join(self.audio_dir, f"{audio_name}.wav"), sr=SR)
            envelope = librosa.onset.onset_strength(data, sr=SR)  # (seq_len,)
            mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
            chroma = librosa.feature.chroma_cens(
                data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T  # (seq_len, 12)

            peak_idxs = librosa.onset.onset_detect(
                onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
            peak_onehot = np.zeros_like(envelope, dtype=np.float32)
            peak_onehot[peak_idxs] = 1.0  # (seq_len,)

            tempo, beat_idxs = librosa.beat.beat_track(
                onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
                start_bpm=_get_tempo(audio_name), tightness=100)
            beat_onehot = np.zeros_like(envelope, dtype=np.float32)
            beat_onehot[beat_idxs] = 1.0  # (seq_len,)

            audio_feature = np.concatenate([
                envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]
            ], axis=-1)
            np.save(save_path, audio_feature)


if __name__ == "__main__":
    dataset = AIChoreoDataset("/mnt/data/AIST++/", "/mnt/data/AIST/music", split="testval")
    print (len(dataset))
    motion, audio, target, seq_name = dataset[0]
    print (motion.shape, audio.shape, target.shape, seq_name)
