import mne
import numpy as np
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

def main():
    # loadin the edf
    raw_A = mne.io.read_raw_edf('Listener.edf', preload=True) 
    # preload for fast access (from cache)
    raw_B = mne.io.read_raw_edf('Speaker.edf', preload=True) 
    # rawEDF → MNE Raw object
    
    events_A, event_id_A = mne.events_from_annotations(raw_A)
    events_B, event_id_B = mne.events_from_annotations(raw_B)
    # looking for the 3 marker 
    # start, end, start (till end)
    din_key = [k for k in event_id_A.keys() if 'DIN1' in k][0]
    din_id = event_id_A[din_key]

    din_events_A = events_A[events_A[:, 2] == din_id] 
    din_events_B = events_B[events_B[:, 2] == din_id]

    sfreq = raw_A.info['sfreq']  
    # get sampling interval
    times_A = din_events_A[:, 0] / sfreq 
    # convert to time
    times_B = din_events_B[:, 0] / sfreq

    pos_A = raw_A.copy().crop(tmin=times_A[0], tmax=times_A[1]) 
    # extract +ve, from start to end (m0 to m1)
    pos_B = raw_B.copy().crop(tmin=times_B[0], tmax=times_B[1])
    neg_A = raw_A.copy().crop(tmin=times_A[2], tmax=None) 
    # negative from m2 to end of recording
    neg_B = raw_B.copy().crop(tmin=times_B[2], tmax=None)

    def process(raw_pos, raw_neg, plot_psd=False):
        raw_combined = mne.concatenate_raws([raw_pos.copy(), raw_neg.copy()]) 
        # JOIN +ve and -ve, not modify org, we alter a copy
        raw_combined.drop_channels(['EEG VREF']) 
        # rem hardware reference electrode data, cuz it impact the average
        raw_combined.filter(1., 40.) 
        # 1 to 40 hz is Brain frequencies 
        raw_combined.set_eeg_reference('average')
        raw_combined.resample(50) 
        # re-referencing as nyquist (2x the freq) -> lesser data

        ica = mne.preprocessing.ICA(n_components=0.95, random_state=42) 
        # SEPERATE to individual comp w 95% variance. 
        ica.fit(raw_combined)
        # instead of manually checking and filtering, we do Kurtosis
        # (gen - smooth oscillations -> brain signals)
        # if more "spiky" could be eye blinks, muscle twitch etc. 
        sources = ica.get_sources(raw_combined).get_data()
        bad_idx = np.where(kurtosis(sources, axis=1) > 40)[0]
        ica.exclude = bad_idx # the ones deemed *not brain*
        print(f"Rejected components: {bad_idx}") 


        def apply_pipeline(raw):
            r = raw.copy() 
            # not touching the original 
            r.drop_channels(['EEG VREF']) 
            r.filter(1., 40.)
            r.set_eeg_reference('average')
            r.resample(50)
            return ica.apply(r) 
        # ICA is already fitted before we use this. 

        if plot_psd: # runs for A only. 
            clean = apply_pipeline(raw_pos)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # before, raw_pos before ICA
            raw_for_psd = raw_pos.copy() 
            raw_for_psd.drop_channels(['EEG VREF'])
            raw_for_psd.filter(1., 40.) 
            # cleaned signgal (all same excpet NO ICA)
            raw_for_psd.set_eeg_reference('average')
            raw_for_psd.resample(50)
            raw_for_psd.compute_psd(fmax=24).plot(axes=axes[0], show=False)
            axes[0].set_title('Before ICA')
            
            # after
            clean.compute_psd(fmax=24).plot(axes=axes[1], show=False)
            axes[1].set_title('After ICA') 
            # on ICA cleaned signal. 
            
            plt.suptitle('Power Spectrum — Participant A', fontsize=13)
            plt.tight_layout()
            plt.savefig('psd_comparison.png', dpi=150)
            plt.show()
            print("PSD saved → psd_comparison.png")

        return apply_pipeline(raw_pos).get_data().T, apply_pipeline(raw_neg).get_data().T

    pos_A_data, neg_A_data = process(pos_A, neg_A, plot_psd=True)
    pos_B_data, neg_B_data = process(pos_B, neg_B, plot_psd=False)

    np.savez('cleaned_data.npz',
             pos_A=pos_A_data, neg_A=neg_A_data,
             pos_B=pos_B_data, neg_B=neg_B_data)
    print("Done. Saved to cleaned_data.npz")

if __name__ == "__main__":
    main()
