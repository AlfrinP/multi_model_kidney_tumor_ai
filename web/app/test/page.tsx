"use client";

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, UploadCloud, Scan, Microchip, Layers, ArrowLeft, CheckCircle } from "lucide-react";
import Link from "next/link";
import styles from "./test.module.css";

export default function TestSystem() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);
  
  const [patientId, setPatientId] = useState("");
  const [mriFile, setMriFile] = useState<File | null>(null);
  const [ctFile, setCtFile] = useState<File | null>(null);

  const mriInputRef = useRef<HTMLInputElement>(null);
  const ctInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent, type: "mri" | "ct") => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (type === "mri") setMriFile(file);
      else setCtFile(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: "mri" | "ct") => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (type === "mri") setMriFile(file);
      else setCtFile(file);
    }
  };

  const handleAnalyze = async () => {
    if (!mriFile || !ctFile) {
      alert("Please upload both MRI and CT scans");
      return;
    }

    setIsAnalyzing(true);
    
    try {
      const formData = new FormData();
      formData.append("mri_file", mriFile);
      formData.append("ct_file", ctFile);

      const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      const response = await fetch(`${apiUrl}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("API request failed");
      }

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error(error);
      alert("Error analyzing scans. Make sure the backend API is running.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <Link href="/" style={{display: 'inline-flex', alignItems: 'center', gap:'0.5rem', color:'var(--text-muted)', marginBottom:'1rem'}}>
          <ArrowLeft size={16} /> Back to Home
        </Link>
        <h1 className={styles.title}><Microchip size={32} color="var(--accent-primary)" /> AI Inference Dashboard</h1>
        <p className={styles.subtitle}>Upload patient CT and MRI scans for multi-modal analysis</p>
      </header>

      <div className={styles.layout}>
        {/* Sidebar */}
        <div className={styles.sidebar}>
          <div className={styles.card}>
            <h2 className={styles.cardTitle}><Activity size={20} /> Patient Data Input</h2>
            
            <div className={styles.inputGroup}>
              <label className={styles.label}>Patient Scan ID</label>
              <input 
                type="text" 
                className={styles.input} 
                placeholder="e.g. PT-10024" 
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
              />
            </div>

            <div className={styles.inputGroup}>
              <label className={styles.label}>T1-Weighted MRI Scan</label>
              <div 
                className={styles.uploadZone}
                onDragOver={handleDragOver}
                onDrop={(e) => handleDrop(e, "mri")}
                onClick={() => mriInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  hidden 
                  ref={mriInputRef} 
                  accept="image/png, image/jpeg, image/jpg"
                  onChange={(e) => handleFileChange(e, "mri")} 
                />
                {mriFile ? (
                  <div style={{display:'flex', flexDirection:'column', alignItems:'center', gap:'0.5rem', color:'var(--success)'}}>
                    <CheckCircle size={28} />
                    <span style={{fontSize:'0.875rem', maxWidth:'100%', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap'}}>{mriFile.name}</span>
                  </div>
                ) : (
                  <>
                    <UploadCloud className={styles.uploadIcon} size={28} />
                    <div className={styles.uploadText}>Click or drag to upload MRI</div>
                  </>
                )}
              </div>
            </div>

            <div className={styles.inputGroup}>
              <label className={styles.label}>Contrast CT Scan</label>
              <div 
                className={styles.uploadZone}
                onDragOver={handleDragOver}
                onDrop={(e) => handleDrop(e, "ct")}
                onClick={() => ctInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  hidden 
                  ref={ctInputRef} 
                  accept="image/png, image/jpeg, image/jpg"
                  onChange={(e) => handleFileChange(e, "ct")} 
                />
                {ctFile ? (
                  <div style={{display:'flex', flexDirection:'column', alignItems:'center', gap:'0.5rem', color:'var(--success)'}}>
                    <CheckCircle size={28} />
                    <span style={{fontSize:'0.875rem', maxWidth:'100%', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap'}}>{ctFile.name}</span>
                  </div>
                ) : (
                  <>
                    <UploadCloud className={styles.uploadIcon} size={28} />
                    <div className={styles.uploadText}>Click or drag to upload CT</div>
                  </>
                )}
              </div>
            </div>

            <button 
              className={styles.analyzeBtn} 
              onClick={handleAnalyze}
              disabled={isAnalyzing || !mriFile || !ctFile}
              style={{opacity: (isAnalyzing || !mriFile || !ctFile) ? 0.7 : 1}}
            >
              <Scan size={20} />
              {isAnalyzing ? "Processing Fusion Model..." : "Analyze Scans"}
            </button>
          </div>
        </div>

        {/* Main Content View */}
        <div className={styles.mainContent}>
          <AnimatePresence mode="wait">
            {!results && !isAnalyzing ? (
              <motion.div 
                key="waiting"
                className={styles.card} 
                style={{textAlign:'center', padding:'6rem 2rem', height:'100%', display:'flex', flexDirection:'column', justifyContent:'center', alignItems:'center'}}
                initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
              >
                <Layers size={48} color="var(--border-strong)" style={{marginBottom:'1.5rem'}} />
                <h3 style={{fontSize:'1.5rem', marginBottom:'0.5rem'}}>Waiting for Input</h3>
                <p style={{color:'var(--text-muted)'}}>Upload coregistered scans and click analyze to view the diagnostic report.</p>
              </motion.div>
            ) : isAnalyzing ? (
              <motion.div 
                key="analyzing"
                className={styles.card} 
                style={{textAlign:'center', padding:'6rem 2rem', height:'100%', display:'flex', flexDirection:'column', justifyContent:'center', alignItems:'center'}}
                initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
              >
                <div style={{position:'relative', width:'80px', height:'80px', marginBottom:'2rem'}}>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                    style={{ position:'absolute', inset:0, border:'3px solid transparent', borderTopColor:'var(--accent-primary)', borderRadius:'50%' }}
                  />
                  <motion.div
                    animate={{ rotate: -360 }}
                    transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                    style={{ position:'absolute', inset:'10px', border:'3px solid transparent', borderTopColor:'var(--accent-secondary)', borderRadius:'50%' }}
                  />
                  <Scan size={32} style={{position:'absolute', top:'50%', left:'50%', transform:'translate(-50%, -50%)', color:'var(--text-main)'}} />
                </div>
                <h3 style={{fontSize:'1.5rem', marginBottom:'0.5rem'}}>Running U-Net Inference...</h3>
                <p style={{color:'var(--accent-primary)'}}>Aggregating spatial & contrast gradients</p>
              </motion.div>
            ) : (
              <motion.div 
                key="results"
                initial={{opacity:0, y:20}} 
                animate={{opacity:1, y:0}} 
                transition={{duration:0.5}}
                style={{display:'flex', flexDirection:'column', gap:'2rem'}}
              >
                <div className={styles.metricsGrid}>
                  <div className={styles.metricCard} style={{borderColor: (results.diagnosis === "No Tumor Detected") ? 'var(--success)' : 'var(--error)', background: (results.diagnosis === "No Tumor Detected") ? 'rgba(34, 197, 94, 0.05)' : 'rgba(239, 68, 68, 0.05)'}}>
                    <span className={styles.metricLabel}>Diagnosis</span>
                    <span className={styles.metricValue} style={{color: (results.diagnosis === "No Tumor Detected") ? 'var(--success)' : 'var(--error)'}}>{results.diagnosis}</span>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>AI Confidence</span>
                    <span className={styles.metricValue} style={{color:'var(--success)'}}>{results.confidence}</span>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>Tumor Volume</span>
                    <span className={styles.metricValue}>{results.volume}</span>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>Max Diameter</span>
                    <span className={styles.metricValue}>{results.diameter}</span>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>Predicted Stage</span>
                    <span className={styles.metricValue} style={{color:'var(--warning)'}}>{results.stage}</span>
                  </div>
                </div>

                <div className={styles.card} style={{borderColor: 'var(--accent-primary)', background: 'rgba(59, 130, 246, 0.05)'}}>
                  <h3 style={{fontSize:'1.125rem', marginBottom:'0.5rem', display:'flex', alignItems:'center', gap:'0.5rem'}}>
                    <Activity size={18} color="var(--accent-primary)" /> Anatomical Localization
                  </h3>
                  <p style={{fontSize:'1.125rem', fontWeight:'600'}}>{results.localization}</p>
                </div>

                <div className={styles.imagesGrid}>

                  <div className={styles.imageContainer}>
                    <div className={styles.imageLabel}>CT Scan Image</div>
                    <div style={{width:'100%', height:'100%', borderRadius:'0.5rem', overflow:'hidden', position:'relative'}}>
                      {results.ctBase64 && (
                        <img src={`data:image/jpeg;base64,${results.ctBase64}`} alt="CT Scan" style={{width:'100%', height:'100%', objectFit:'contain'}} />
                      )}
                    </div>
                  </div>
                  <div className={styles.imageContainer}>
                    <div className={styles.imageLabel}>CT AI Mask</div>
                    <div style={{width:'100%', height:'100%', borderRadius:'0.5rem', overflow:'hidden', position:'relative'}}>
                      {results.ctMaskBase64 && (
                        <img src={`data:image/jpeg;base64,${results.ctMaskBase64}`} alt="CT Mask" style={{width:'100%', height:'100%', objectFit:'contain'}} />
                      )}
                    </div>
                  </div>

                  <div className={styles.imageContainer}>
                    <div className={styles.imageLabel}>MRI Scan Image</div>
                    <div style={{width:'100%', height:'100%', borderRadius:'0.5rem', overflow:'hidden', position:'relative'}}>
                      {results.mriBase64 && (
                        <img src={`data:image/jpeg;base64,${results.mriBase64}`} alt="MRI Scan" style={{width:'100%', height:'100%', objectFit:'contain'}} />
                      )}
                    </div>
                  </div>
                  <div className={styles.imageContainer}>
                    <div className={styles.imageLabel}>MRI AI Mask</div>
                    <div style={{width:'100%', height:'100%', borderRadius:'0.5rem', overflow:'hidden', position:'relative'}}>
                      {results.mriMaskBase64 && (
                        <img src={`data:image/jpeg;base64,${results.mriMaskBase64}`} alt="MRI Mask" style={{width:'100%', height:'100%', objectFit:'contain'}} />
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
