require "xlua";
require "string";

do
   local function getTermLength()
      if sys.uname() == 'windows' then return 80 end
      local tputf = io.popen('tput cols', 'r')
      local w = tonumber(tputf:read('*a'))
      local rc = {tputf:close()}
      if rc[3] == 0 then return w
      else return 80 end 
   end

   local barDone = true
   local previous = -1
   local tm = ''
   local timer
   local times
   local indices
   local termLength = math.min(getTermLength(), 110)

   function progress(current, goal, cost)
      -- Defaults:
      local barLength = termLength - 12
      local smoothing = 100 
      local maxfps = 10
      
      -- Compute percentage
      local percent = math.floor(((current) * barLength) / goal)

      -- Start new bar
      if (barDone and ((previous == -1) or (percent < previous))) then
         barDone = false
         previous = -1
         tm = ''
         timer = torch.Timer()
         times = {timer:time().real}
         indices = {current}
      else
         io.write('\r')
      end

      -- If (percent ~= previous and not barDone) then
      if (not barDone) then
         previous = percent
         -- Print bar
         io.write(' [')
         for i=1,barLength do
            if (i < percent) then io.write('=')
            elseif (i == percent) then io.write('>')
            else io.write('.') end
         end
         io.write('] ')
         for i=1,termLength-barLength-4 do io.write(' ') end
         for i=1,termLength-barLength-4 do io.write('\b') end
         -- Time stats
         local elapsed = timer:time().real
         local step = (elapsed-times[1]) / (current-indices[1])
         if current==indices[1] then step = 0 end
         local remaining = math.max(0,(goal - current)*step)
         table.insert(indices, current)
         table.insert(times, elapsed)
         if #indices > smoothing then
            indices = table.splice(indices)
            times = table.splice(times)
         end
         -- Print remaining time when running or total time when done.
         if (percent < barLength) then
            io.write('ETA: ' .. xlua.formatTime(remaining))
         else
            io.write('Tot: ' .. xlua.formatTime(elapsed))
         end
         io.write(' | Step: ' .. xlua.formatTime(step))

		 io.write(' | Cost: ' .. string.format("%.2f", cost))
		 
         -- Go back to center of bar, and print progress
         for i=1,6+#tm+barLength/2 do io.write('\b') end
         io.write(' ', current, '/', goal, ' ')
         -- Reset for next bar
         if (percent == barLength) then
            barDone = true
            io.write('\n')
         end
         -- Flush
         io.write('\r')
         io.flush()
      end
   end
end
