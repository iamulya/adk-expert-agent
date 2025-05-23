"use client"; 

import { motion } from "framer-motion";
import { cn } from "~/lib/utils";

export function Welcome({ className }: { className?: string }) {
  return (
    <motion.div
      className={cn("flex flex-col items-center text-center", className)} // Added text-center
      style={{ transition: "all 0.2s ease-out" }}
      initial={{ opacity: 0, scale: 0.85, y: 20 }} // Added initial y offset
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }} // Slightly adjusted transition
    >
      <div className="text-5xl mb-4" role="img" aria-label="Robot Waving">🤖👋</div> {/* Robot emoji */}
      <h3 className="mb-3 text-2xl font-semibold md:text-3xl">
        ADK Expert Agent
      </h3>
      <p className="text-muted-foreground max-w-md text-sm md:text-base">
        Hello! I'm here to help you with your queries about Google's Agent Development Kit (ADK).
        I can answer any ADK related question, even provide you guidance about any of ADK's GitHub issues. 
        Furthermore, I can output the information you request in pdf, slides and can also generate architecture diagrams.
        <br />
        <b>Important: </b>Since the best model is being used to give you the most accurate and detailed answers, it can take some time to respond to certain queries where deeper analysis is required.
        <br />
      </p>
    </motion.div>
  );
}